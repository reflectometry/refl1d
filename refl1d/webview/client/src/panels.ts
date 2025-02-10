import { shared_state } from "bumps-webview-client/src/app_state";
import { panels as bumps_panels, type Panel } from "bumps-webview-client/src/panels";
import DataView from "./components/DataView.vue";
import ModelView from "./components/ModelView.vue";
import ProfileUncertaintyView from "./components/ProfileUncertaintyView.vue";
import SimpleBuilder from "./components/SimpleBuilder.vue";

const refl1dPanels: Panel[] = [
  { title: "Reflectivity", component: DataView },
  { title: "Profile", component: ModelView },
  {
    title: "Profile Uncertainty",
    component: ProfileUncertaintyView,
    show: () => shared_state.uncertainty_available?.available ?? false,
  },
  { title: "Builder", component: SimpleBuilder },
];

const replacements = {
  Reflectivity: "Data",
  "Profile Uncertainty": "Model Uncertainty",
};

const insertions = {
  Profile: { after: "Summary" },
  Builder: { after: "Uncertainty" },
};

function replace_panel(
  panels: Panel[],
  replacement_panels: Panel[],
  replaced_title: string,
  replacement_title: string
) {
  const index = panels.findIndex((p) => p.title === replaced_title);
  const replacement_index = replacement_panels.findIndex((p) => p.title === replacement_title);
  if (index >= 0 && replacement_index >= 0) {
    panels.splice(index, 1, replacement_panels[replacement_index]);
  }
}

function insert_panel(panels: Panel[], insertion_panels: Panel[], title: string, after: string) {
  let index = panels.findIndex((p) => p.title === after);
  // put it at the end if the after panel is not found
  if (index < 0) {
    index = panels.length - 1;
  }
  const insertion_index = insertion_panels.findIndex((p) => p.title === title);
  if (insertion_index >= 0) {
    panels.splice(index + 1, 0, insertion_panels[insertion_index]);
  }
}

export const panels = [...bumps_panels];
for (const [replacement_title, replaced_title] of Object.entries(replacements)) {
  replace_panel(panels, refl1dPanels, replaced_title, replacement_title);
}
for (const [title, { after }] of Object.entries(insertions)) {
  insert_panel(panels, refl1dPanels, title, after);
}
