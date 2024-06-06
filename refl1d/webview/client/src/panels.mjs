import { panels as bumps_panels } from 'bumps-webview-client/src/panels';
import DataView from './components/DataView.vue';
import ModelView from './components/ModelView.vue';
import SimpleBuilder from './components/SimpleBuilder.vue';
import ProfileUncertaintyView from './components/ProfileUncertaintyView.vue';

const refl1d_panels = [
    {title: 'Reflectivity', component: DataView},
    {title: 'Profile', component: ModelView},
    {title: 'Profile Uncertainty', component: ProfileUncertaintyView},
    {title: 'Builder', component: SimpleBuilder},
];

const replacements = {
    'Reflectivity': 'Data',
    'Profile Uncertainty': 'Model Uncertainty',
}

const insertions = {
    'Profile': { after: 'Summary'},
    'Builder': { after: 'Uncertainty' }
}

function replace_panel(panels, replacement_panels, replaced_title, replacement_title) {
    const index = panels.findIndex(p => p.title === replaced_title);
    const replacement_index = replacement_panels.findIndex(p => p.title === replacement_title);
    if (index >= 0 && replacement_index >= 0) {
        panels.splice(index, 1, replacement_panels[replacement_index]);
    }
}

function insert_panel(panels, insertion_panels, title, after) {
    let index = panels.findIndex(p => p.title === after);
    // put it at the end if the after panel is not found
    if (index < 0) {
        index = panels.length - 1;
    }
    const insertion_index = insertion_panels.findIndex(p => p.title === title);
    if (insertion_index >= 0) {
        panels.splice(index+1, 0, insertion_panels[insertion_index]);
    }
}

export const panels = [...bumps_panels];
for (const [replacement_title, replaced_title] of Object.entries(replacements)) {
    replace_panel(panels, refl1d_panels, replaced_title, replacement_title);
}
for (const [title, {after}] of Object.entries(insertions)) {
    insert_panel(panels, refl1d_panels, title, after);
}
