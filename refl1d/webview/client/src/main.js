import { createApp } from 'vue'
import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css'
import App from 'bumps-webview-client/src/App.vue';
import { panels } from './panels.mjs';

createApp(App, {panels}).mount('#app');