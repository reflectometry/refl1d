import pickle
from copy import deepcopy

import dill
import cloudpickle

from bumps.serialize import deserialize, serialize

from refl1d.names import SLD, Slab


def test_stack_serialization():
    """test that stack can be serialized and deserialized with all the methods we use,
    preserving the functioning of the Calculation object for the total thickness"""
    sample = Slab(SLD(rho=10), thickness=10) | Slab(SLD(rho=10), thickness=20) | Slab(SLD(rho=10), thickness=30)
    thickness_plus = sample.thickness + 100  # expression

    ser_t, ser_s = deserialize(serialize([thickness_plus, sample]))
    assert ser_t.value == 160
    ser_s[0].thickness.value += 40
    assert ser_t.value == 200

    dc_t, dc_s = deepcopy([thickness_plus, sample])
    assert dc_t.value == 160
    dc_s[0].thickness.value += 40
    assert dc_t.value == 200

    pickle_t, pickle_s = pickle.loads(pickle.dumps([thickness_plus, sample]))
    assert pickle_t.value == 160
    pickle_s[0].thickness.value += 40
    assert pickle_t.value == 200

    dill_t, dill_s = dill.loads(dill.dumps([thickness_plus, sample]))
    assert dill_t.value == 160
    dill_s[0].thickness.value += 40
    assert dill_t.value == 200

    cloudpickle_t, cloudpickle_s = cloudpickle.loads(cloudpickle.dumps([thickness_plus, sample]))
    assert cloudpickle_t.value == 160
    cloudpickle_s[0].thickness.value += 40
    assert cloudpickle_t.value == 200

    assert thickness_plus.value == 160
    sample[0].thickness.value += 40
    assert thickness_plus.value == 200


def test_repeat_serialization():
    """test that stack can be serialized and deserialized with all the methods we use,
    preserving the functioning of the Calculation object for the total thickness"""
    unit_cell = Slab(SLD(rho=10), thickness=10) | Slab(SLD(rho=10), thickness=20) | Slab(SLD(rho=10), thickness=30)
    # This creates a Repeat object from the Stack:
    sample = unit_cell * 4
    thickness_plus = sample.thickness + 100  # expression

    ser_t, ser_s = deserialize(serialize([thickness_plus, sample]))
    assert ser_t.value == 340  # (10+20+30 * 4) + 100
    ser_s.stack[0].thickness.value += 40
    assert ser_t.value == 500  # (50+20+30 * 4) + 100

    dc_t, dc_s = deepcopy([thickness_plus, sample])
    assert dc_t.value == 340
    dc_s.stack[0].thickness.value += 40
    assert dc_t.value == 500

    pickle_t, pickle_s = pickle.loads(pickle.dumps([thickness_plus, sample]))
    assert pickle_t.value == 340
    pickle_s.stack[0].thickness.value += 40
    assert pickle_t.value == 500

    dill_t, dill_s = dill.loads(dill.dumps([thickness_plus, sample]))
    assert dill_t.value == 340
    dill_s.stack[0].thickness.value += 40
    assert dill_t.value == 500

    cloudpickle_t, cloudpickle_s = cloudpickle.loads(cloudpickle.dumps([thickness_plus, sample]))
    assert cloudpickle_t.value == 340
    cloudpickle_s.stack[0].thickness.value += 40
    assert cloudpickle_t.value == 500

    assert thickness_plus.value == 340
    sample.stack[0].thickness.value += 40
    assert thickness_plus.value == 500
