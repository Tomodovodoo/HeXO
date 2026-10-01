"""Child processes that end together with everything they start."""
import contextlib
import ctypes
import os
import signal
import subprocess
import time


class _Limits(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION."""
    _fields_ = [('times', ctypes.c_int64 * 2), ('flags', ctypes.c_uint32), ('working_set', ctypes.c_size_t * 2),
                ('processes', ctypes.c_uint32), ('affinity', ctypes.c_size_t), ('priority', ctypes.c_uint32),
                ('scheduling', ctypes.c_uint32), ('io', ctypes.c_uint64 * 6), ('memory', ctypes.c_size_t * 4)]


def _kernel32():
    kernel32 = ctypes.windll.kernel32
    handle, flag = ctypes.c_void_p, ctypes.c_int
    for name, result, arguments in (
            ('CreateJobObjectW', handle, [handle, handle]),
            ('SetInformationJobObject', flag, [handle, ctypes.c_int, handle, ctypes.c_uint32]),
            ('QueryInformationJobObject', flag, [handle, ctypes.c_int, handle, ctypes.c_uint32, handle]),
            ('AssignProcessToJobObject', flag, [handle, handle]),
            ('TerminateJobObject', flag, [handle, ctypes.c_uint]),
            ('CloseHandle', flag, [handle])):
        getattr(kernel32, name).restype = result
        getattr(kernel32, name).argtypes = arguments
    return kernel32


class TreeProcess(subprocess.Popen):
    """A Popen whose kill() ends the process and every process it started. On Windows they share a job object,
    which also ends them when this process exits or when the leader is reaped; elsewhere they share a session, which
    is killed when the leader is killed or reaped."""

    def __init__(self, args, **kwargs):
        self.job, self.session_ended = None, False
        if os.name != 'nt':
            super().__init__(args, start_new_session=True, **kwargs)
            return
        self.kernel32 = _kernel32()
        self.job = self.kernel32.CreateJobObjectW(None, None)
        limits = _Limits(flags=0x2000)   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            if not self.job or not self.kernel32.SetInformationJobObject(self.job, 9, ctypes.byref(limits),
                                                                         ctypes.sizeof(limits)):
                raise OSError('Unable to create a job object')
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(args, **kwargs)
        except BaseException:
            self._release()
            raise
        if not self.kernel32.AssignProcessToJobObject(self.job, int(self._handle)):
            self.kill()
            self.wait()
            raise OSError('Unable to put a process in its job object')

    def _running(self):
        """How many processes of the job are alive."""
        counts = (ctypes.c_uint32 * 12)()   # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION: four times, then four counts
        self.kernel32.QueryInformationJobObject(self.job, 1, counts, ctypes.sizeof(counts), None)
        return counts[10]

    def _release(self):
        if self.job:
            self.kernel32.CloseHandle(self.job)
            self.job = None

    def _end_session(self):
        if not self.session_ended:
            self.session_ended = True
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.pid, signal.SIGKILL)

    def kill(self):
        if self.job:
            self.kernel32.TerminateJobObject(self.job, 1)
            deadline = time.monotonic() + 5
            while self._running() and time.monotonic() < deadline:
                time.sleep(.01)
            self._release()
        elif os.name != 'nt':
            self._end_session()
        super().kill()

    terminate = kill

    def wait(self, timeout=None):
        """Reap the process and end what it left running: closing the job on Windows, signalling the session
        elsewhere."""
        code = super().wait(timeout)
        self._release()
        if os.name != 'nt':
            self._end_session()
        return code
