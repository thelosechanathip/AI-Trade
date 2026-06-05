import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import run


class NpmDiscoveryTests(unittest.TestCase):
    @patch("run.shutil.which")
    def test_find_npm_uses_path(self, which: Mock) -> None:
        which.return_value = r"C:\Tools\nodejs\npm.cmd"

        self.assertEqual(run._find_npm(), r"C:\Tools\nodejs\npm.cmd")

    @patch("run.os.path.isfile")
    @patch("run.shutil.which", return_value=None)
    def test_find_npm_uses_standard_windows_path(
        self, _which: Mock, isfile: Mock
    ) -> None:
        isfile.side_effect = lambda path: path == r"C:\Program Files\nodejs\npm.cmd"

        with (
            patch.object(run.sys, "platform", "win32"),
            patch.dict(
                run.os.environ,
                {"ProgramFiles": r"C:\Program Files"},
                clear=True,
            ),
        ):
            self.assertEqual(
                run._find_npm(),
                r"C:\Program Files\nodejs\npm.cmd",
            )


class NpmInstallTests(unittest.TestCase):
    @patch("run.subprocess.run")
    def test_install_dependencies_reports_success(self, subprocess_run: Mock) -> None:
        subprocess_run.return_value = Mock(returncode=0)

        self.assertTrue(run._install_nextjs_dependencies("dashboard", "npm.cmd"))

    @patch("run.subprocess.run", side_effect=FileNotFoundError("npm.cmd"))
    def test_install_dependencies_handles_missing_npm(
        self, _subprocess_run: Mock
    ) -> None:
        self.assertFalse(run._install_nextjs_dependencies("dashboard", "npm.cmd"))

    @patch("run.subprocess.run")
    def test_install_dependencies_handles_nonzero_exit(
        self, subprocess_run: Mock
    ) -> None:
        subprocess_run.return_value = Mock(returncode=1)

        self.assertFalse(run._install_nextjs_dependencies("dashboard", "npm.cmd"))


if __name__ == "__main__":
    unittest.main()
