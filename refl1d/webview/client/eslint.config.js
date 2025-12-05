import url from "node:url";
import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import vueTsEslintConfig from "@vue/eslint-config-typescript";
import pluginVue from "eslint-plugin-vue";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  // allConfig: js.configs.all,
});

export default [
  /** Extend recommended configs */
  ...compat.extends("plugin:vue/recommended", "plugin:vuejs-accessibility/recommended"),
  ...pluginVue.configs["flat/recommended"],
  ...vueTsEslintConfig(),
  /** Configuration */
  {
    languageOptions: {
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "script",
      },
    },
    files: ["src/**/*.js", "src/**/*.mjs", "src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
    /** Override rules */
    rules: {
      "max-len": ["error", { code: 120 }],
      "prefer-const": 0,
      "@typescript-eslint/ban-ts-comment": ["error", { "ts-ignore": "allow-with-description" }],
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-expressions": [
        "error",
        { allowShortCircuit: true, allowTernary: true }, // Temporary fix for indirect dependency @typescript-eslint <= 8.15.0
      ],
      "vuejs-accessibility/label-has-for": [
        "error",
        {
          required: {
            some: ["nesting", "id"],
          },
        },
      ],
      "vue/max-attributes-per-line": [
        "error",
        {
          singleline: {
            max: 5,
          },
        },
      ],
      "vue/html-self-closing": [
        "error",
        {
          html: {
            void: "any",
            normal: "any",
          },
        },
      ],
    },
  },
];
