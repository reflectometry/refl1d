<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
// import { Modal } from 'bootstrap';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import { isArray } from '@vue/shared';

const props = defineProps({
  fitter_defaults: {
    type: Object,
    default: {}
  },
  fitter_settings: {
    type: Object,
    default: {}
  },
  active_fitter: {
    type: String,
    default: 'amoeba'
  }
});

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'update', value: Object): void
  (e: 'active-fitter', value: string): void
  (e: 'active-settings', value: Object): void
}>();

const dialog = ref(null);
const isOpen = ref(false);

const FIT_FIELDS = {
  'starts': ['Starts', 'integer'],
  'steps': ['Steps', 'integer'],
  'samples': ['Samples', 'integer'],
  'xtol': ['x tolerance', 'float'],
  'ftol': ['f(x) tolerance', 'float'],
  'alpha': ['Convergence', 'float'],
  'stop': ['Stopping criteria', 'string'],
  'thin': ['Thinning', 'integer'],
  'burn': ['Burn-in steps', 'integer'],
  'pop': ['Population', 'float'],
  'init': ['Initializer', ["eps", "lhs", "cov", "random"]],
  'CR': ['Crossover ratio', 'float'],
  'F': ['Scale', 'float'],
  'nT': ['# Temperatures', 'integer'],
  'Tmin': ['Min temperature', 'float'],
  'Tmax': ['Max temperature', 'float'],
  'radius': ['Simplex radius', 'float'],
  'trim': ['Burn-in trim', 'boolean'],
  'outliers': ['Outliers', ["none", "iqr", "grubbs", "mahal"]]
}

const active_fitter = ref('amoeba');
// work with a copy of the fitter definitions, which will
// still get used as defaults

// make another working copy for editing:
const active_settings = ref({});

let modal: Modal;

onMounted(() => {
  modal = new Modal(dialog.value, { backdrop: 'static', keyboard: false });
});

function close() {
  modal?.hide();
}

function open() {
  // copy the active_fitter from the server state:
  active_fitter.value = props.active_fitter;
  changeActiveFitter();
  modal?.show();
}

const fit_names = computed(() => Object.keys(props.fitter_defaults));

function changeActiveFitter() {
  active_settings.value = structuredClone(props.fitter_settings[active_fitter.value]?.settings);
}

// watch(isOpen, (newState, oldState) => {
//   if (newState == true && modal) {
//     modal.show();
//   }
//   else {
//     modal.hide();
//   }
// })

// watch(() => props.fitter_settings, (newState, oldState) => {
//   // fit_names.value = Object.keys(newState);
// })

function process_settings() {
  return Object.fromEntries(Object.entries(active_settings.value).map(([sname, value]) => {
    const field_type = FIT_FIELDS[sname][1];
    let processed_value = value;
    if (field_type === 'integer') {
      processed_value = Math.round(Number(value));
    }
    else if (field_type === 'float') {
      processed_value = Number(value);
    }
    else if (field_type === 'boolean') {
      // probably unnecessary if it is bound to a checkbox
      processed_value = Boolean(value);
    }
    return [sname, processed_value];
  }))
}

function save() {
  const new_settings = {};
  new_settings[active_fitter.value] = {settings: process_settings()};
  emit('active-settings', new_settings);
  emit('active-fitter', active_fitter.value);
  close();
}

function reset() {
  active_settings.value = structuredClone(props.fitter_defaults[props.active_fitter].settings);
}

function validate(value, field_name) {
  const field_type = FIT_FIELDS[field_name][1];
  if (isArray(field_type) || field_type === 'boolean') {
    // there's no way to get an incorrect option.
    return true;
  }
  const float_value = Number(value);
  if (isNaN(float_value)) {
    return false;
  }
  if (field_type === 'integer' && parseInt(value, 10) != float_value) {
    return false;
  }
  return true;
}

const anyIsInvalid = computed(() => {
  return Object.entries(active_settings.value).some(([sname, value]) => !validate(value, sname));
})

defineExpose({
  close,
  open
})
</script>

<template>
  <div ref="dialog" class="modal fade" id="fitOptionsModal" tabindex="-1" aria-labelledby="fitOptionsLabel"
    :aria-hidden="isOpen">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="fitOptionsLabel">Fit Options</h5>
          <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="container">
            <div class="row border-bottom">
              <div class="col">
                <div class="form-check" v-for="fname in fit_names.slice(0,3)" :key="fname">
                  <input class="form-check-input" v-model="active_fitter" type="radio" name="flexRadio" :id="fname"
                    :value="fname" @change="changeActiveFitter">
                  <label class="form-check-label" :for="fname">
                    {{props.fitter_defaults[fname].name}}
                  </label>
                </div>
              </div>
              <div class="col">
                <div class="form-check" v-for="fname in fit_names.slice(3)" :key="fname">
                  <input class="form-check-input" v-model="active_fitter" type="radio" name="flexRadio" :id="fname"
                    :value="fname" @change="changeActiveFitter">
                  <label class="form-check-label" :for="fname">
                    {{props.fitter_defaults[fname].name}}
                  </label>
                </div>
              </div>
            </div>
            <div class="row p-2">
              <div class="row p-1" v-for="(value, sname, index) in active_settings" :key="sname">
                <label class="col-sm-4 col-form-label" :for="'setting_' + index">{{FIT_FIELDS[sname][0]}}</label>
                <div class="col-sm-8">
                  <select v-if="isArray(FIT_FIELDS[sname][1])" v-model="active_settings[sname]" class="form-select">
                    <option v-for="opt in FIT_FIELDS[sname][1]">{{opt}}</option>
                  </select>
                  <input v-else-if="FIT_FIELDS[sname][1]==='boolean'" class="form-check-input m-2" type="checkbox" v-model="active_settings[sname]" />
                  <input v-else :class="{'form-control': true, 'is-invalid': !validate(active_settings[sname], sname)}" type="text" v-model="active_settings[sname]" /></div>
              </div>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" @click="reset">Reset Defaults</button>
          <button type="button" class="btn btn-primary" :class="{disabled: anyIsInvalid}" @click="save">Save Changes</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.form-check label {
  user-select: none;
}
</style>