import { panels as bumps_panels } from 'bumps-webview-client/src/panels';
import DataView from './components/DataView.vue';
import ModelView from './components/ModelView.vue';

export const panels = [...bumps_panels];

// replace the bumps data view with the refl1d one.
panels.find(p => p.title === 'Data').component = DataView;
// insert the profile panel at position 3
panels.splice(2, 0, {title: 'Profile', component: ModelView});
