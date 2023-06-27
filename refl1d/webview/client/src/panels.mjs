import { panels as bumps_panels } from 'bumps-webview-client/src/panels';
import DataView from './components/DataView.vue';
import ModelView from './components/ModelView.vue';
import ProfileUncertaintyView from './components/ProfileUncertaintyView.vue';

export const panels = [...bumps_panels];

// replace the bumps data view with the refl1d one.
panels.splice(0, 1, {title: 'Reflectivity', component: DataView});
// insert the profile panel at position 3
panels.splice(2, 0, {title: 'Profile', component: ModelView});
panels.splice(9, 1, {title: 'Profile Uncertainty', component: ProfileUncertaintyView});
