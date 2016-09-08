import json
from unittest import TestCase

from ect.core.op import OpMetaInfo
from ect.core.workflow import Workflow
from ect.ui.workspace import Workspace


class WorkspaceTest(TestCase):
    def test_example(self):
        expected_json_text = """{
            "qualified_name": "workspace_workflow",
            "header": {
                "description": "Test!"
            },
            "input": {},
            "output": {
                "p": {
                    "data_type": "xarray.core.dataset.Dataset",
                    "source": "p.return"
                },
                "ts": {
                    "data_type": "xarray.core.dataset.Dataset",
                    "source": "ts.return",
                    "description": "A timeseries dataset."
                }
            },
            "steps": [
                {
                    "id": "p",
                    "op": "ect.ops.io.read_netcdf",
                    "input": {
                        "file": {
                            "value": "2010_precipitation.nc"
                        },
                        "drop_variables": {},
                        "decode_cf": {},
                        "decode_times": {},
                        "engine": {}
                    },
                    "output": {
                        "return": {}
                    }
                },
                {
                    "id": "ts",
                    "op": "ect.ops.timeseries.timeseries",
                    "input": {
                        "ds": {
                            "source": "p.return"
                        },
                        "lat": {
                            "value": 53
                        },
                        "lon": {
                            "value": 10
                        },
                        "method": {}
                    },
                    "output": {
                        "return": {}
                    }
                }
            ]
        }
        """

        expected_json_dict = json.loads(expected_json_text)

        ws = Workspace('/path', Workflow(OpMetaInfo('workspace_workflow', header_dict=dict(description='Test!'))))
        # print("wf_1: " + json.dumps(ws.workflow.to_json_dict(), indent='  '))
        ws.add_resource('p', 'ect.ops.io.read_netcdf', ["file=2010_precipitation.nc"])
        # print("wf_2: " + json.dumps(ws.workflow.to_json_dict(), indent='  '))
        ws.add_resource('ts', 'ect.ops.timeseries.timeseries', ["ds=p", "lat=53", "lon=10"])
        print("wf_3: " + json.dumps(ws.workflow.to_json_dict(), indent='  '))
        self.assertEqual(ws.workflow.to_json_dict(), expected_json_dict)
