"""Tests for the local UI's HTTP API.

The key property under test is that the UI and the MCP server are two doors onto
one backend: a job queued through the API is visible to the MCP tools, and one
queued by an MCP tool is visible and cancellable from the API.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.jobs.store import JobStore
from ffmpeg_mcp.models import JobStatus
from ffmpeg_mcp.ui.media import parse_range
from ffmpeg_mcp.ui.server import create_app

from .helpers import call_tool

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A UI client with no auth token, as when bound to loopback."""
    app = create_app(settings, token=None)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def secured_client(settings: Settings) -> Iterator[TestClient]:
    """A UI client that requires a token, as when bound to a LAN address."""
    app = create_app(settings, token="secret-token")
    with TestClient(app) as test_client:
        yield test_client


class TestHealthAndWorkspace:
    def test_health_reports_the_workspace_and_tool_count(
        self, client: TestClient, settings: Settings
    ) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["workspace"] == str(settings.workspace)
        assert body["tool_count"] > 25
        assert body["auth_required"] is False

    def test_workspace_lists_the_allowed_roots(
        self, client: TestClient, settings: Settings
    ) -> None:
        body = client.get("/api/workspace").json()
        assert body["workspace"] == str(settings.workspace)
        assert str(settings.workspace) in body["allowed_roots"]


class TestAuth:
    def test_health_is_reachable_without_a_token(self, secured_client: TestClient) -> None:
        assert secured_client.get("/api/health").status_code == 200

    def test_data_endpoints_require_a_token(self, secured_client: TestClient) -> None:
        assert secured_client.get("/api/jobs").status_code == 401

    def test_a_valid_header_token_is_accepted(self, secured_client: TestClient) -> None:
        response = secured_client.get("/api/jobs", headers={"x-auth-token": "secret-token"})
        assert response.status_code == 200

    def test_a_valid_query_token_is_accepted(self, secured_client: TestClient) -> None:
        assert secured_client.get("/api/jobs?token=secret-token").status_code == 200

    def test_a_wrong_token_is_rejected(self, secured_client: TestClient) -> None:
        response = secured_client.get("/api/jobs", headers={"x-auth-token": "nope"})
        assert response.status_code == 401


class TestTools:
    def test_every_tool_is_listed_with_a_schema(self, client: TestClient) -> None:
        tools = client.get("/api/tools").json()
        names = {t["name"] for t in tools}
        assert {"trim", "resize_video", "render_timeline", "auto_caption"} <= names
        for tool in tools:
            assert tool["description"]
            assert tool["input_schema"]["type"] == "object"

    def test_a_read_only_tool_can_be_called_directly(self, client: TestClient, clip: Path) -> None:
        response = client.post("/api/tools/probe_media", json={"input_path": str(clip)})
        assert response.status_code == 200
        assert response.json()["duration"] == pytest.approx(3.0, abs=0.2)

    def test_calling_an_unknown_tool_is_a_404(self, client: TestClient) -> None:
        assert client.post("/api/tools/nope", json={}).status_code == 404

    def test_invalid_arguments_return_a_validation_error(
        self, client: TestClient, clip: Path
    ) -> None:
        response = client.post(
            "/api/tools/trim", json={"input_path": str(clip), "start": 5.0, "end": 1.0}
        )
        assert response.status_code == 422

    def test_a_path_outside_the_allowlist_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/tools/probe_media", json={"input_path": "/etc/hosts"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_path"


class TestJobs:
    def test_a_job_queued_through_the_api_appears_in_the_list(
        self, client: TestClient, clip: Path
    ) -> None:
        submission = client.post(
            "/api/tools/trim", json={"input_path": str(clip), "start": 0.0, "end": 1.0}
        ).json()
        listing = client.get("/api/jobs").json()
        assert submission["job_id"] in {job["job_id"] for job in listing["jobs"]}
        # The UI's worker pool may already have claimed it, so assert on the
        # job's presence rather than on it still sitting in the queue.
        entry = next(j for j in listing["jobs"] if j["job_id"] == submission["job_id"])
        assert entry["tool"] == "trim"
        assert sum(listing["counts"].values()) >= 1

    async def test_a_job_queued_by_an_mcp_tool_is_visible_to_the_api(
        self, client: TestClient, clip: Path
    ) -> None:
        # This is the point of sharing one store between the two front doors.
        submission = await call_tool("trim", {"input_path": str(clip), "start": 0.0, "end": 1.0})
        response = client.get(f"/api/jobs/{submission['job_id']}")
        assert response.status_code == 200
        assert response.json()["tool"] == "trim"

    def test_job_detail_includes_params_and_status(self, client: TestClient, clip: Path) -> None:
        submission = client.post(
            "/api/tools/trim", json={"input_path": str(clip), "start": 0.5, "end": 1.5}
        ).json()
        body = client.get(f"/api/jobs/{submission['job_id']}").json()
        # The UI runs its own worker pool, so a freshly queued job may already
        # have started. What matters here is that the request was recorded
        # faithfully and the job is in some valid lifecycle state.
        assert body["params"]["start"] == 0.5
        assert body["params"]["end"] == 1.5
        assert body["status"] in {"queued", "running", "done"}

    def test_an_unknown_job_is_an_error(self, client: TestClient) -> None:
        assert client.get("/api/jobs/does-not-exist").status_code == 400

    def test_filtering_by_status(self, client: TestClient, clip: Path) -> None:
        # The UI's own worker pool is running, so a queued job may start at any
        # moment; filter on a settled state rather than racing it.
        import time

        submission = client.post(
            "/api/tools/trim", json={"input_path": str(clip), "end": 1.0}
        ).json()
        deadline = time.time() + 60
        while time.time() < deadline:
            if client.get(f"/api/jobs/{submission['job_id']}").json()["status"] == "done":
                break
            time.sleep(0.2)
        done = client.get("/api/jobs?status=done").json()["jobs"]
        assert submission["job_id"] in {job["job_id"] for job in done}
        assert client.get("/api/jobs?status=cancelled").json()["jobs"] == []

    def test_cancelling_a_queued_job_through_the_api(self, client: TestClient, clip: Path) -> None:
        submission = client.post(
            "/api/tools/trim", json={"input_path": str(clip), "end": 1.0}
        ).json()
        body = client.post(f"/api/jobs/{submission['job_id']}/cancel").json()
        assert body["status"] == "cancelled"

    async def test_the_api_can_cancel_a_job_the_mcp_side_queued(
        self, client: TestClient, clip: Path, settings: Settings
    ) -> None:
        import asyncio

        submission = await call_tool("trim", {"input_path": str(clip), "end": 1.0})
        client.post(f"/api/jobs/{submission['job_id']}/cancel")
        # Cancellation is cooperative, so a job the UI pool already started ends
        # cancelled shortly afterwards rather than instantly.
        for _ in range(100):
            status = await call_tool("job_status", {"job_id": submission["job_id"]})
            if status["status"] in {"cancelled", "done", "failed"}:
                break
            await asyncio.sleep(0.1)
        assert status["status"] == "cancelled"

    def test_the_ui_worker_pool_runs_a_job_to_completion(
        self, client: TestClient, clip: Path, settings: Settings
    ) -> None:
        # The UI must be usable with no MCP client attached at all.
        import time

        submission = client.post(
            "/api/tools/trim",
            json={"input_path": str(clip), "start": 0.0, "end": 1.0, "mode": "reencode"},
        ).json()
        job_id = submission["job_id"]
        deadline = time.time() + 60
        body = {}
        while time.time() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] in {"done", "failed", "cancelled"}:
                break
            time.sleep(0.2)
        assert body["status"] == "done", body
        assert Path(body["outputs"][0]).exists()


class TestFiles:
    def test_workspace_media_is_listed(
        self, client: TestClient, clip: Path, settings: Settings
    ) -> None:
        files = client.get("/api/files").json()
        assert str(clip) in {f["path"] for f in files}
        entry = next(f for f in files if f["path"] == str(clip))
        assert entry["kind"] == "video"
        assert entry["size_bytes"] > 0

    def test_non_media_files_are_skipped(self, client: TestClient, settings: Settings) -> None:
        (settings.workspace / "notes.xyz").write_text("hi")
        assert all(not f["name"].endswith(".xyz") for f in client.get("/api/files").json())

    def test_listing_outside_the_allowlist_is_rejected(self, client: TestClient) -> None:
        assert client.get("/api/files?directory=/etc").status_code == 400


class TestMedia:
    def test_a_whole_file_is_served(self, client: TestClient, clip: Path) -> None:
        response = client.get(f"/api/media?path={clip}")
        assert response.status_code == 200
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["content-type"] == "video/mp4"

    def test_a_range_request_returns_partial_content(self, client: TestClient, clip: Path) -> None:
        # Without this, the browser's <video> element cannot seek.
        response = client.get(f"/api/media?path={clip}", headers={"Range": "bytes=0-1023"})
        assert response.status_code == 206
        assert response.headers["content-length"] == "1024"
        assert response.headers["content-range"].startswith("bytes 0-1023/")
        assert len(response.content) == 1024

    def test_an_open_ended_range_runs_to_the_end(self, client: TestClient, clip: Path) -> None:
        size = clip.stat().st_size
        response = client.get(f"/api/media?path={clip}", headers={"Range": "bytes=100-"})
        assert response.status_code == 206
        assert int(response.headers["content-length"]) == size - 100

    def test_serving_outside_the_allowlist_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/media?path=/etc/hosts")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_path"

    def test_a_missing_file_is_rejected(self, client: TestClient, settings: Settings) -> None:
        response = client.get(f"/api/media?path={settings.workspace / 'nope.mp4'}")
        assert response.status_code == 400


class TestRangeParsing:
    def test_a_bounded_range(self) -> None:
        assert parse_range("bytes=0-499", 1000) == (0, 499)

    def test_an_open_ended_range(self) -> None:
        assert parse_range("bytes=500-", 1000) == (500, 999)

    def test_a_suffix_range_returns_the_tail(self) -> None:
        assert parse_range("bytes=-200", 1000) == (800, 999)

    def test_an_end_past_the_file_is_clamped(self) -> None:
        assert parse_range("bytes=0-99999", 1000) == (0, 999)

    def test_a_start_past_the_file_is_unsatisfiable(self) -> None:
        assert parse_range("bytes=2000-", 1000) is None

    def test_an_inverted_range_is_rejected(self) -> None:
        assert parse_range("bytes=500-100", 1000) is None

    @pytest.mark.parametrize("header", ["", "bytes=", "items=0-10", "garbage"])
    def test_malformed_headers_fall_back_to_the_whole_file(self, header: str) -> None:
        assert parse_range(header, 1000) is None


class TestWebSocket:
    def test_job_updates_are_pushed(self, client: TestClient, clip: Path) -> None:
        with client.websocket_connect("/ws/jobs") as socket:
            submission = client.post(
                "/api/tools/trim",
                json={"input_path": str(clip), "start": 0.0, "end": 1.0, "mode": "reencode"},
            ).json()
            seen: set[str] = set()
            for _ in range(40):
                message = socket.receive_json()
                assert message["type"] == "jobs"
                for job in message["jobs"]:
                    if job["job_id"] == submission["job_id"]:
                        seen.add(job["status"])
                if "done" in seen or "failed" in seen:
                    break
            assert "done" in seen, f"never saw completion, saw {seen}"

    def test_the_socket_requires_a_token_when_one_is_set(self, secured_client: TestClient) -> None:
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect), secured_client.websocket_connect("/ws/jobs"):
            pass

    def test_the_socket_accepts_a_valid_token(self, secured_client: TestClient) -> None:
        with secured_client.websocket_connect("/ws/jobs?token=secret-token") as socket:
            assert socket is not None


class TestCrossProcessVisibility:
    def test_a_job_written_by_another_store_instance_shows_up(
        self, client: TestClient, settings: Settings, clip: Path
    ) -> None:
        # Stands in for the MCP server running as a separate process.
        other = JobStore(settings)
        record = other.create("trim", {"input_path": str(clip), "end": 1.0})
        body = client.get(f"/api/jobs/{record.job_id}").json()
        assert body["job_id"] == record.job_id
        assert body["status"] in {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.DONE.value,
        }


def test_the_ui_does_not_need_the_mcp_server_running(
    settings: Settings, sample_video: Path
) -> None:
    """The UI is startable and usable on its own."""
    shutil.copy(sample_video, settings.workspace / "standalone.mp4")
    app = create_app(settings, token=None)
    with TestClient(app) as standalone:
        assert standalone.get("/api/health").json()["status"] == "ok"
        assert standalone.get("/api/tools").status_code == 200
        assert len(standalone.get("/api/files").json()) >= 1


class TestUpload:
    def test_a_file_is_taken_into_the_workspace(
        self, client: TestClient, settings: Settings, sample_video: Path
    ) -> None:
        response = client.post(
            "/api/upload",
            files={"files": ("holiday.mp4", sample_video.read_bytes(), "video/mp4")},
        )
        assert response.status_code == 200
        entry = response.json()[0]
        assert Path(entry["path"]).parent == settings.uploads_dir
        assert entry["kind"] == "video"
        assert entry["size_bytes"] == sample_video.stat().st_size

    def test_an_uploaded_file_is_immediately_usable_by_a_tool(
        self, client: TestClient, sample_video: Path
    ) -> None:
        # The whole point: media from outside the allowed roots becomes editable.
        uploaded = client.post(
            "/api/upload",
            files={"files": ("holiday.mp4", sample_video.read_bytes(), "video/mp4")},
        ).json()[0]
        probe = client.post("/api/tools/probe_media", json={"input_path": uploaded["path"]})
        assert probe.status_code == 200
        assert probe.json()["duration"] == pytest.approx(3.0, abs=0.2)

    def test_an_uploaded_file_shows_up_in_the_file_listing(
        self, client: TestClient, sample_video: Path
    ) -> None:
        uploaded = client.post(
            "/api/upload", files={"files": ("clip.mp4", sample_video.read_bytes(), "video/mp4")}
        ).json()[0]
        assert uploaded["path"] in {f["path"] for f in client.get("/api/files").json()}

    def test_several_files_upload_at_once(
        self, client: TestClient, sample_video: Path, sample_image: Path
    ) -> None:
        response = client.post(
            "/api/upload",
            files=[
                ("files", ("a.mp4", sample_video.read_bytes(), "video/mp4")),
                ("files", ("b.png", sample_image.read_bytes(), "image/png")),
            ],
        )
        assert [entry["kind"] for entry in response.json()] == ["video", "image"]

    def test_a_traversing_filename_cannot_escape_the_uploads_directory(
        self, client: TestClient, settings: Settings, sample_video: Path
    ) -> None:
        response = client.post(
            "/api/upload",
            files={"files": ("../../../../tmp/pwned.mp4", sample_video.read_bytes(), "video/mp4")},
        )
        assert response.status_code == 200
        written = Path(response.json()[0]["path"])
        assert written.parent == settings.uploads_dir
        assert not Path("/tmp/pwned.mp4").exists()

    def test_an_executable_extension_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/upload", files={"files": ("evil.sh", b"#!/bin/sh\nrm -rf /", "text/x-sh")}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_parameter"

    def test_an_oversized_upload_is_refused(self, workspace: Path, sample_video: Path) -> None:
        from ..conftest import make_settings

        tight = make_settings(workspace, max_input_bytes=64)
        tight.ensure_dirs()
        with TestClient(create_app(tight, token=None)) as tight_client:
            response = tight_client.post(
                "/api/upload",
                files={"files": ("big.mp4", sample_video.read_bytes(), "video/mp4")},
            )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "file_too_large"
            assert list(tight.uploads_dir.iterdir()) == []

    def test_upload_requires_a_token_when_one_is_set(
        self, secured_client: TestClient, sample_video: Path
    ) -> None:
        response = secured_client.post(
            "/api/upload", files={"files": ("a.mp4", sample_video.read_bytes(), "video/mp4")}
        )
        assert response.status_code == 401

    def test_the_accepted_types_are_advertised(self, client: TestClient) -> None:
        body = client.get("/api/upload/accepts").json()
        assert ".mp4" in body["suffixes"]
        assert ".sh" not in body["suffixes"]
        assert body["max_bytes"] > 0


class TestProjectScoping:
    """Two sessions on one server should not bury each other's jobs."""

    def test_jobs_can_be_listed_per_project(
        self, client: TestClient, settings: Settings, clip: Path
    ) -> None:
        from ffmpeg_mcp.jobs.store import JobStore

        other = JobStore(settings)
        other.create("trim", {"input_path": str(clip)}, project="wedding")
        other.create("trim", {"input_path": str(clip)}, project="reel")
        other.create("trim", {"input_path": str(clip)}, project="reel")

        reel = client.get("/api/jobs?project=reel").json()
        assert reel["total"] == 2
        assert {j["project"] for j in reel["jobs"]} == {"reel"}
        assert client.get("/api/jobs?project=wedding").json()["total"] == 1

    def test_all_spans_projects(self, client: TestClient, settings: Settings, clip: Path) -> None:
        from ffmpeg_mcp.jobs.store import JobStore

        other = JobStore(settings)
        other.create("trim", {}, project="a")
        other.create("trim", {}, project="b")
        assert client.get("/api/jobs?project=all").json()["total"] >= 2

    def test_paging_reports_whether_more_remains(
        self, client: TestClient, settings: Settings
    ) -> None:
        from ffmpeg_mcp.jobs.store import JobStore

        other = JobStore(settings)
        for _ in range(5):
            other.create("trim", {}, project="paged")

        first = client.get("/api/jobs?project=paged&limit=2&offset=0").json()
        assert len(first["jobs"]) == 2
        assert first["total"] == 5
        assert first["has_more"] is True

        last = client.get("/api/jobs?project=paged&limit=2&offset=4").json()
        assert len(last["jobs"]) == 1
        assert last["has_more"] is False

    def test_projects_are_listed_with_counts(self, client: TestClient, settings: Settings) -> None:
        from ffmpeg_mcp.jobs.store import JobStore

        JobStore(settings).create("trim", {}, project="listed")
        rows = {p["name"]: p for p in client.get("/api/projects").json()}
        assert rows["listed"]["jobs"] == 1
