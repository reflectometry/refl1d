import { autoupdate_state } from "bumps-webview-client/src/app_state";
import { panels as bumpsPanels } from "bumps-webview-client/src/panels";
import { afterEach, expect, test, vi } from "vitest";
import DataPanel from "../../src/components/DataView.vue";
import ModelView from "../../src/components/ModelView.vue";
import ProfileUncertaintyView from "../../src/components/ProfileUncertaintyView.vue";
import SimpleBuilder from "../../src/components/SimpleBuilder.vue";
import { panels } from "../../src/panels";

vi.mock("plotly.js/lib/core", () => import("./plotlyMock"));

// Refl1D's panels are spliced into the bumps list by title, so a renamed bumps
// panel would silently put a Refl1D panel in the wrong place, or drop it.
const titles = () => panels.map((p) => p.title);
const after = (title: string) => titles()[titles().indexOf(title) + 1];

afterEach(() => {
  autoupdate_state.shared_state.uncertainty_available.value = undefined;
});

test("Reflectivity replaces the bumps Data panel, as the first tab", () => {
  expect(panels[0]).toMatchObject({ title: "Reflectivity", component: DataPanel });
  expect(titles()).not.toContain("Data");
});

test("Profile Uncertainty replaces the bumps Model Uncertainty panel", () => {
  const index = bumpsPanels.findIndex((p) => p.title === "Model Uncertainty");
  expect(index).toBeGreaterThan(-1);
  expect(panels.find((p) => p.title === "Profile Uncertainty")?.component).toBe(ProfileUncertaintyView);
  expect(titles()).not.toContain("Model Uncertainty");
});

test("Profile follows Summary, and Builder follows Uncertainty", () => {
  expect(after("Summary")).toBe("Profile");
  expect(after("Uncertainty")).toBe("Builder");
  expect(panels.find((p) => p.title === "Profile")?.component).toBe(ModelView);
  expect(panels.find((p) => p.title === "Builder")?.component).toBe(SimpleBuilder);
});

test("Every bumps panel is kept, and no panel appears twice", () => {
  expect(new Set(titles()).size).toBe(titles().length);
  const replaced = ["Data", "Model Uncertainty"];
  for (const { title } of bumpsPanels.filter((p) => !replaced.includes(p.title))) {
    expect(titles()).toContain(title);
  }
  expect(panels).toHaveLength(bumpsPanels.length + 2);
});

test("Profile Uncertainty is only shown once a fit has uncertainty", () => {
  const panel = panels.find((p) => p.title === "Profile Uncertainty")!;
  expect(panel.show?.()).toBe(false);
  autoupdate_state.shared_state.uncertainty_available.value = { available: true, num_points: 100 };
  expect(panel.show?.()).toBe(true);
});
