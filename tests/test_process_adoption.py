"""No real bridge, network, or robot; exercise Linux pipe handover."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quest_process_adoption import AdoptedInput


class AdoptionTests(unittest.TestCase):
    def test_exact_child_adoption_keeps_pipe_open_after_parent_closes_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'quest_unitree_bridge.py'
            script.write_text('import time\nprint("ready", flush=True)\ntime.sleep(30)\n')
            child = subprocess.Popen([sys.executable, '-u', str(script), '--host', '10.0.0.1'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            adopted = None
            try:
                self.assertEqual(child.stdout.readline(), 'ready\n')
                with self.assertRaises(ValueError):
                    AdoptedInput(child.pid, root, '10.0.0.2')
                with self.assertRaises(ValueError):
                    AdoptedInput(child.pid, root / 'wrong', '10.0.0.1')
                adopted = AdoptedInput(child.pid, root, '10.0.0.1')
                child.stdout.close()
                child.stderr.close()
                self.assertIsNone(adopted.poll())
                adopted.terminate()
                self.assertEqual(adopted.wait(timeout=3), 0)
            finally:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=3)
                for stream in (child.stdout, child.stderr):
                    stream.close()
                if adopted:
                    adopted.stdout.close()
                    adopted.stderr.close()

    def test_invalid_and_unrelated_process_rejected(self):
        for pid in (True, 0, -1, '123', os.getpid()):
            with self.assertRaises(ValueError):
                AdoptedInput(pid, Path.cwd(), '10.0.0.1')
