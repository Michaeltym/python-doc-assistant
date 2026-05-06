/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DOCS_VERSION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
