import { createApp } from 'vue'
import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css'
import App from 'bumps-webview-client/src/App.vue';
import { panels } from './panels.mjs';

const name = "Refl1D";

createApp(App, {panels, name}).mount('#app');