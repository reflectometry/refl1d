import { ref, shallowRef } from 'vue';

// Whether data files to be loaded have their dQ as standard deviation or FWHM
export const dq_is_FWHM = ref(false);