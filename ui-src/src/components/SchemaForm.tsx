/**
 * A form generated from a tool's JSON schema.
 *
 * The schemas come straight from the pydantic models, so the form cannot drift
 * from what the tools accept. Scalars, enums and nested objects render as real
 * controls; anything genuinely structural (a list of timeline clips, a set of
 * curve points) falls back to a JSON editor rather than pretending to be a
 * builder it is not.
 */

import { useState } from "react";
import type { JsonSchema } from "../api";

export type Values = Record<string, unknown>;

function resolve(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  const name = schema.$ref.replace("#/$defs/", "");
  return root.$defs?.[name] ?? schema;
}

/** Unwrap `anyOf: [X, null]`, which is how pydantic renders an optional field. */
function simplify(schema: JsonSchema, root: JsonSchema): { schema: JsonSchema; optional: boolean } {
  const resolved = resolve(schema, root);
  if (resolved.anyOf) {
    const variants = resolved.anyOf.map((entry) => resolve(entry, root));
    const nonNull = variants.filter((entry) => entry.type !== "null");
    const optional = variants.length !== nonNull.length;
    if (nonNull.length === 1) {
      return { schema: { ...nonNull[0], description: resolved.description }, optional };
    }
    // A union of several real types (e.g. a Literal set) — collect the enums.
    const enums = nonNull.flatMap((entry) => entry.enum ?? (entry.const ? [entry.const] : []));
    if (enums.length > 0) {
      return { schema: { type: "string", enum: enums, description: resolved.description }, optional };
    }
  }
  if (resolved.allOf?.length === 1) {
    return { schema: resolve(resolved.allOf[0], root), optional: false };
  }
  return { schema: resolved, optional: false };
}

function enumOptions(schema: JsonSchema): unknown[] | null {
  if (schema.enum) return schema.enum;
  return null;
}

interface FieldProps {
  name: string;
  schema: JsonSchema;
  root: JsonSchema;
  required: boolean;
  value: unknown;
  onChange: (value: unknown) => void;
}

function Field({ name, schema: raw, root, required, value, onChange }: FieldProps) {
  const { schema, optional } = simplify(raw, root);
  const label = schema.title || name;
  const hint = schema.description;
  const choices = enumOptions(schema);

  if (choices) {
    return (
      <div className="field">
        <label htmlFor={name}>
          {label}
          {required && !optional ? " *" : ""}
        </label>
        <select
          id={name}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) =>
            onChange(event.target.value === "" ? undefined : event.target.value)
          }
        >
          <option value="">{optional || !required ? "(default)" : "Choose…"}</option>
          {choices.map((choice) => (
            <option key={String(choice)} value={String(choice)}>
              {String(choice)}
            </option>
          ))}
        </select>
        {hint && <span className="hint">{hint}</span>}
      </div>
    );
  }

  if (schema.type === "boolean") {
    return (
      <div className="field">
        <label className="row" htmlFor={name}>
          <input
            id={name}
            type="checkbox"
            checked={Boolean(value ?? schema.default ?? false)}
            onChange={(event) => onChange(event.target.checked)}
          />
          <span>{label}</span>
        </label>
        {hint && <span className="hint">{hint}</span>}
      </div>
    );
  }

  if (schema.type === "number" || schema.type === "integer") {
    return (
      <div className="field">
        <label htmlFor={name}>
          {label}
          {required && !optional ? " *" : ""}
        </label>
        <input
          id={name}
          type="number"
          step={schema.type === "integer" ? 1 : "any"}
          min={schema.minimum ?? schema.exclusiveMinimum}
          max={schema.maximum ?? schema.exclusiveMaximum}
          placeholder={schema.default !== undefined ? String(schema.default) : ""}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => {
            const text = event.target.value;
            onChange(text === "" ? undefined : Number(text));
          }}
        />
        {hint && <span className="hint">{hint}</span>}
      </div>
    );
  }

  if (schema.type === "object" && schema.properties) {
    return (
      <details className="field" style={{ display: "block" }}>
        <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--text-dim)" }}>
          {label}
        </summary>
        <div style={{ marginTop: 10, paddingLeft: 12, borderLeft: "1px solid var(--border)" }}>
          <ObjectFields
            schema={schema}
            root={root}
            values={(value as Values) ?? {}}
            onChange={(next) => onChange(Object.keys(next).length ? next : undefined)}
          />
        </div>
      </details>
    );
  }

  if (schema.type === "array") {
    const item = schema.items ? simplify(schema.items, root).schema : undefined;
    const simpleItems = item && item.type !== "object" && !item.properties;
    if (simpleItems) {
      const text = Array.isArray(value) ? (value as unknown[]).join("\n") : "";
      return (
        <div className="field">
          <label htmlFor={name}>
            {label}
            {required ? " *" : ""}
          </label>
          <textarea
            id={name}
            rows={3}
            placeholder="One per line"
            value={text}
            onChange={(event) => {
              const lines = event.target.value
                .split("\n")
                .map((line) => line.trim())
                .filter(Boolean);
              const coerced =
                item?.type === "number" || item?.type === "integer"
                  ? lines.map(Number)
                  : lines;
              onChange(coerced.length ? coerced : undefined);
            }}
          />
          {hint && <span className="hint">{hint}</span>}
        </div>
      );
    }
  }

  // Structural values keep a JSON editor: pretending otherwise would produce a
  // worse builder than just showing the data.
  if (schema.type === "array" || (schema.type === "object" && !schema.properties)) {
    return <JsonField name={name} label={label} hint={hint} value={value} onChange={onChange} />;
  }

  return (
    <div className="field">
      <label htmlFor={name}>
        {label}
        {required && !optional ? " *" : ""}
      </label>
      <input
        id={name}
        type="text"
        placeholder={schema.default !== undefined ? String(schema.default) : ""}
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(event) => onChange(event.target.value === "" ? undefined : event.target.value)}
      />
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

function JsonField({
  name,
  label,
  hint,
  value,
  onChange,
}: {
  name: string;
  label: string;
  hint?: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [text, setText] = useState(() => (value === undefined ? "" : JSON.stringify(value, null, 2)));
  const [invalid, setInvalid] = useState(false);
  return (
    <div className="field">
      <label htmlFor={name}>{label} (JSON)</label>
      <textarea
        id={name}
        rows={6}
        className="mono"
        value={text}
        onChange={(event) => {
          const next = event.target.value;
          setText(next);
          if (next.trim() === "") {
            setInvalid(false);
            onChange(undefined);
            return;
          }
          try {
            onChange(JSON.parse(next));
            setInvalid(false);
          } catch {
            setInvalid(true);
          }
        }}
      />
      {invalid && <span className="hint" style={{ color: "var(--bad)" }}>Not valid JSON yet.</span>}
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

function ObjectFields({
  schema,
  root,
  values,
  onChange,
}: {
  schema: JsonSchema;
  root: JsonSchema;
  values: Values;
  onChange: (values: Values) => void;
}) {
  const required = new Set(schema.required ?? []);
  return (
    <>
      {Object.entries(schema.properties ?? {}).map(([name, property]) => (
        <Field
          key={name}
          name={name}
          schema={property}
          root={root}
          required={required.has(name)}
          value={values[name]}
          onChange={(next) => {
            const updated = { ...values };
            if (next === undefined) delete updated[name];
            else updated[name] = next;
            onChange(updated);
          }}
        />
      ))}
    </>
  );
}

export default function SchemaForm({
  schema,
  values,
  onChange,
}: {
  schema: JsonSchema;
  values: Values;
  onChange: (values: Values) => void;
}) {
  return <ObjectFields schema={schema} root={schema} values={values} onChange={onChange} />;
}
