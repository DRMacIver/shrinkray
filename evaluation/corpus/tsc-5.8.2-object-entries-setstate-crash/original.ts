// A small reactive form/store module: components hold typed state and
// reconcile it from untyped external payloads. tsc 5.8.2 crashes while
// emitting declaration-visibility info for one of the component classes.

export interface Field {
  name: string;
  label: string;
  required: boolean;
}

export type Validator = (value: unknown) => string | null;

export const validators: Record<string, Validator> = {
  nonEmpty: (v) => (typeof v === "string" && v.length > 0 ? null : "required"),
  isNumber: (v) => (typeof v === "number" ? null : "not a number"),
};

export function describeField(field: Field): string {
  const suffix = field.required ? " (required)" : "";
  return `${field.label}${suffix}`;
}

export type State = {
  a: number;
  b: string;
};

export class Store<T extends object> {
  private listeners: Array<(value: T) => void> = [];

  constructor(private value: T) {}

  subscribe(listener: (value: T) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  get(): T {
    return this.value;
  }

  set(next: T): void {
    this.value = next;
    for (const listener of this.listeners) {
      listener(next);
    }
  }
}

export class Test {
  setState(state: State) {}
  test = (e: any) => {
    for (const [key, value] of Object.entries(e)) {
      this.setState({
        [key]: value,
      });
    }
  };
}

export function makeStore(): Store<State> {
  return new Store<State>({ a: 0, b: "" });
}

export function summarize(fields: Field[]): string {
  return fields.map(describeField).join(", ");
}
