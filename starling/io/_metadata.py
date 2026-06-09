"""ID03 BLISS metadata parsing (vendored from the local darling checkout).

Parses scan commands (ascan/fscan/fscan2d/amesh/...) from a BLISS master file
into scan shape, motor h5 paths, integrated-motor flags and the detector data
key. Includes the local beamline patches (amesh support) that are not in
upstream darling.
"""

import h5py
import numpy as np


class ID03:
    """Scan metadata configuration for the ESRF ID03 beamline BLISS format.

    Args:
        abs_path_to_h5_file (str): absolute path to the BLISS master h5 file.
    """

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file

        # argument positions inside the scan command string, per command:
        # e.g. fscan2d motor1 start1 step1 n1 motor2 start2 step2 n2 ...
        self.scan_arg_pos = {
            "motor_steps": {
                "ascan": [3],
                "fscan": [3],
                "a2scan": [6],
                "d2scan": [6, 6],
                "fscan2d": [3, 7],
                "loopscan": [0],
                "amesh": [3, 7],
            },
            "motor_names": {
                "ascan": [0],
                "fscan": [0],
                "a2scan": [0],
                "d2scan": [0, 3],
                "fscan2d": [0, 4],
                "amesh": [0, 4],
            },
        }

        # which motors are integrated (swept during exposure) per command
        self.is_integrated = {
            "ascan": [False],
            "fscan": [True],
            "a2scan": [False],
            "d2scan": [False, False],
            "fscan2d": [False, True],
            "loopscan": [None],
            "amesh": [False, False],
        }

        # scan-command motor name -> h5 storage location
        self.motor_map = {
            "ccmth": "instrument/positioners/ccmth",
            "s8vg": "instrument/positioners/s8vg",
            "s8vo": "instrument/positioners/s8vo",
            "s8ho": "instrument/positioners/s8ho",
            "s8hg": "instrument/positioners/s8hg",
            "chi": "instrument/chi/value",
            "phi": "instrument/phi/value",
            "mu": "instrument/positioners/mu",
            "diffrz": "instrument/diffrz/data",
            "diffry": "instrument/diffry/data",
            "diffrx": "instrument/diffrx/data",
            "omega": "instrument/positioners/omega",
            "ux": "instrument/positioners/ux",
            "uy": "instrument/positioners/uy",
            "uz": "instrument/positioners/uz",
            "mainx": "instrument/positioners/mainx",
            "obx": "instrument/positioners/obx",
            "oby": "instrument/positioners/oby",
            "obz": "instrument/positioners/obz",
            "obz3": "instrument/positioners/obz3",
            "obpitch": "instrument/positioners/obpitch",
            "obyaw": "instrument/positioners/obyaw",
            "sovg": "instrument/positioners/sovg",
            "sovo": "instrument/positioners/sovo",
            "soho": "instrument/positioners/soho",
            "sohg": "instrument/positioners/sohg",
            "cdx": "instrument/positioners/cdx",
            "dcx": "instrument/positioners/dcx",
            "dcz": "instrument/positioners/dcz",
            "ffz": "instrument/positioners/ffz",
            "ffy": "instrument/positioners/ffy",
            "ffsel": "instrument/positioners/ffsel",
            "x_pixel_size": "instrument/pco_ff/x_pixel_size",
            "y_pixel_size": "instrument/pco_ff/y_pixel_size",
        }

        self.fallback_motor_map = {
            self.motor_map["mu"]: "instrument/mu/data",
            self.motor_map["chi"]: "instrument/positioners/chi",
            self.motor_map["phi"]: "instrument/positioners/phi",
            self.motor_map["soho"]: "instrument/soho/value",
            self.motor_map["ux"]: "instrument/positioners/samx",
            self.motor_map["uy"]: "instrument/positioners/samy",
            self.motor_map["uz"]: "instrument/positioners/samz",
            self.motor_map["omega"]: "instrument/omega/data",
        }

        # independent sensors stored under the .2 layer (1.2, 2.2, ...)
        self.sensor_names = {
            "pico4": "instrument/pico4/data",
            "pico3": "instrument/pico3/data",
            "current": "instrument/current/data",
            "elapsed_time": "instrument/elapsed_time/value",
        }

    def __call__(self, scan_id):
        """Parse all scan parameters and sensor data for a scan id.

        Returns:
            tuple: scan_params dict (scan_command, scan_shape, motor_names,
            integrated_motors, data_name, scan_id, invariant_motors) and a
            sensors dict.
        """
        scan_params = {}
        scan_params["scan_command"] = self._get_scan_command(scan_id)
        scan_params["scan_shape"] = self._get_scan_shape(scan_params)
        scan_params["motor_names"] = self._get_motor_names(
            scan_params, scan_id, scan_params["scan_shape"]
        )
        scan_params["integrated_motors"] = self._get_integrated_motors(scan_params)
        scan_params["data_name"] = self._get_data_name(
            scan_id, scan_params["scan_shape"]
        )
        scan_params["scan_id"] = scan_id
        scan_params["invariant_motors"] = self._get_invariant_motors(
            scan_params["motor_names"], scan_id
        )
        sensors = self._get_sensor_data(scan_id)
        return scan_params, sensors

    def _get_scan_command(self, scan_id):
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            return h5f[scan_id]["title"][()].decode("utf-8")

    def _get_scan_shape(self, scan_params):
        command = scan_params["scan_command"].split(" ")[0]
        params = np.array(scan_params["scan_command"].split(" ")[1:])
        scan_shape = params[self.scan_arg_pos["motor_steps"][command]].astype(int)
        # interval-style commands count intervals, not points
        if command in ("a2scan", "ascan", "amesh"):
            scan_shape = scan_shape + 1
        return scan_shape

    def _get_motor_names(self, scan_params, scan_id, scan_shape):
        command = scan_params["scan_command"].split(" ")[0]
        params = scan_params["scan_command"].split(" ")[1:]

        if command == "loopscan":
            return None

        trial_motor_names = [
            self.motor_map[params[i]] for i in self.scan_arg_pos["motor_names"][command]
        ]

        motor_names = []
        expected_number_of_frames = np.prod(scan_shape)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            for motor_name in trial_motor_names:
                fallback = self.fallback_motor_map.get(motor_name)
                if (
                    motor_name in h5f[scan_id]
                    and h5f[scan_id][motor_name].size == expected_number_of_frames
                ):
                    motor_names.append(motor_name)
                elif (
                    fallback is not None
                    and fallback in h5f[scan_id]
                    and h5f[scan_id][fallback].size == expected_number_of_frames
                ):
                    motor_names.append(fallback)
                else:
                    raise ValueError(
                        f"Could not find {motor_name} with fallback name {fallback}"
                    )
        return motor_names

    def _get_integrated_motors(self, scan_params):
        command = scan_params["scan_command"].split(" ")[0]
        return self.is_integrated[command]

    def _get_invariant_motors(self, moving_motor_names, scan_id):
        invariant_motors = {}
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            for motor_key, h5_motor_path in self.motor_map.items():
                if moving_motor_names is None or h5_motor_path not in moving_motor_names:
                    fallback = self.fallback_motor_map.get(h5_motor_path)
                    if h5_motor_path in h5f[scan_id]:
                        invariant_motors[motor_key] = h5f[scan_id][h5_motor_path][()]
                    elif fallback is not None and fallback in h5f[scan_id]:
                        invariant_motors[motor_key] = h5f[scan_id][fallback][()]
        return invariant_motors

    def _get_sensor_data(self, scan_id):
        sensor_scan_id = scan_id.split(".")[0] + ".2"
        sensors = {}
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            for sensor_name, h5_sensor_path in self.sensor_names.items():
                if sensor_scan_id in h5f and h5_sensor_path in h5f[sensor_scan_id]:
                    sensors[sensor_name] = h5f[sensor_scan_id][h5_sensor_path][()]
                else:
                    sensors[sensor_name] = None
        return sensors

    def _get_data_name(self, scan_id, scan_shape):
        """The h5 key of the detector image stack: the 3D dataset whose first
        dimension matches the total number of scan points."""
        leafs = []
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            h5f[scan_id].visititems(lambda name, obj: leafs.append(name))
            while leafs:
                leaf = leafs.pop()
                if (
                    isinstance(h5f[scan_id][leaf], h5py.Dataset)
                    and len(h5f[scan_id][leaf].shape) == 3
                    and h5f[scan_id][leaf].shape[0] == np.prod(scan_shape)
                ):
                    return leaf
        raise ValueError("No dataset found in h5 file")
