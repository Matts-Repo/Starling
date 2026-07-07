"""ID03 BLISS metadata parsing.

Parses scan commands (ascan/fscan/fscan2d/amesh/...) from a BLISS master file
into scan shape, motor h5 paths, integrated-motor flags and the detector data
key. Includes local beamline patches (amesh support).
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

    def __call__(self, scan_id, h5f=None):
        """Parse all scan parameters and sensor data for a scan id.

        All metadata (title parse, motor resolution, detector discovery,
        invariant motors, sensors) is served from a single ``h5py.File``
        open — repeated open/close cycles are an NFS/GPFS latency bomb on
        beamline filesystems. Pass an already-open ``h5f`` handle to fold
        this call into an enclosing open (e.g. a stacked-scan metadata pass).

        Returns:
            tuple: scan_params dict (scan_command, scan_shape, motor_names,
            integrated_motors, data_name, scan_id, invariant_motors) and a
            sensors dict.
        """
        if h5f is None:
            with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
                return self(scan_id, h5f=h5f)

        scan_params = {}
        scan_params["scan_command"] = self._get_scan_command(scan_id, h5f)
        scan_params["scan_shape"] = self._get_scan_shape(scan_params)
        scan_params["motor_names"] = self._get_motor_names(
            scan_params, scan_id, scan_params["scan_shape"], h5f
        )
        scan_params["integrated_motors"] = self._get_integrated_motors(scan_params)
        scan_params["data_name"] = self._get_data_name(
            scan_id, scan_params["scan_shape"], h5f
        )
        scan_params["scan_id"] = scan_id
        scan_params["invariant_motors"] = self._get_invariant_motors(
            scan_params["motor_names"], scan_id, h5f
        )
        sensors = self._get_sensor_data(scan_id, h5f)
        return scan_params, sensors

    def _get_scan_command(self, scan_id, h5f):
        return h5f[scan_id]["title"][()].decode("utf-8")

    def _get_scan_shape(self, scan_params):
        command = scan_params["scan_command"].split(" ")[0]
        params = np.array(scan_params["scan_command"].split(" ")[1:])
        scan_shape = params[self.scan_arg_pos["motor_steps"][command]].astype(int)
        # interval-style commands count intervals, not points
        if command in ("a2scan", "ascan", "amesh"):
            scan_shape = scan_shape + 1
        return scan_shape

    def _get_motor_names(self, scan_params, scan_id, scan_shape, h5f):
        command = scan_params["scan_command"].split(" ")[0]
        params = scan_params["scan_command"].split(" ")[1:]

        if command == "loopscan":
            return None

        trial_motor_names = [
            self.motor_map[params[i]] for i in self.scan_arg_pos["motor_names"][command]
        ]

        motor_names = []
        expected_number_of_frames = np.prod(scan_shape)

        scan_group = h5f[scan_id]
        for motor_name in trial_motor_names:
            fallback = self.fallback_motor_map.get(motor_name)
            if (
                motor_name in scan_group
                and scan_group[motor_name].size == expected_number_of_frames
            ):
                motor_names.append(motor_name)
            elif (
                fallback is not None
                and fallback in scan_group
                and scan_group[fallback].size == expected_number_of_frames
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

    def _get_invariant_motors(self, moving_motor_names, scan_id, h5f):
        invariant_motors = {}
        scan_group = h5f[scan_id]
        for motor_key, h5_motor_path in self.motor_map.items():
            if moving_motor_names is None or h5_motor_path not in moving_motor_names:
                fallback = self.fallback_motor_map.get(h5_motor_path)
                if h5_motor_path in scan_group:
                    invariant_motors[motor_key] = scan_group[h5_motor_path][()]
                elif fallback is not None and fallback in scan_group:
                    invariant_motors[motor_key] = scan_group[fallback][()]
        return invariant_motors

    def _get_sensor_data(self, scan_id, h5f):
        sensor_scan_id = scan_id.split(".")[0] + ".2"
        sensors = {}
        sensor_group = h5f[sensor_scan_id] if sensor_scan_id in h5f else None
        for sensor_name, h5_sensor_path in self.sensor_names.items():
            if sensor_group is not None and h5_sensor_path in sensor_group:
                sensors[sensor_name] = sensor_group[h5_sensor_path][()]
            else:
                sensors[sensor_name] = None
        return sensors

    def _get_data_name(self, scan_id, scan_shape, h5f):
        """The h5 key of the detector image stack: the 3D dataset whose first
        dimension matches the total number of scan points."""
        n_expected = np.prod(scan_shape)
        matches = []

        def visit(name, obj):
            # test on the visited object itself: re-opening h5f[scan_id][name]
            # per leaf churns hundreds of h5py object constructions per scan
            if (
                isinstance(obj, h5py.Dataset)
                and len(obj.shape) == 3
                and obj.shape[0] == n_expected
            ):
                matches.append(name)

        h5f[scan_id].visititems(visit)
        if matches:
            # the legacy scan popped leafs from the end of the visit list, so
            # keep returning the last visited match
            return matches[-1]
        raise ValueError("No dataset found in h5 file")
