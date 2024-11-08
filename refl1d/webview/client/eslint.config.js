import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import prettierConfig from "@vue/eslint-config-prettier";
import vueTsEslintConfig from "@vue/eslint-config-typescript";
import eslintPluginPrettierRecommended from "eslint-plugin-prettier/recommended";
import pluginVue from "eslint-plugin-vue";
import url from "url";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  // allConfig: js.configs.all,
});

export default [
  /** Extend recommended configs */
  ...compat.extends(
    "plugin:vue/vue3-recommended",
    "plugin:vuejs-accessibility/recommended",
    "prettier"
  ),
  ...pluginVue.configs["flat/recommended"],
  ...vueTsEslintConfig(),
  eslintPluginPrettierRecommended,
  prettierConfig,
  /** Configuration */
  {
    languageOptions: {
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "script",
      },
    },
    files: ["src/**/*.ts", "src/**/*.tsx"],
    /** Override rules */
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "max-len": ["error", { code: 120 }],
      "prettier/prettier": [
        "warn",
        {},
        {
          usePrettierrc: true,
        },
      ],
      "react/no-unescaped-entities": "off",
    },
  },
];