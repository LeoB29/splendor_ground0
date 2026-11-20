"""Bounded raw-pipe chunk reads with unchanged trusted protocol semantics.

The frozen parent creates stdout as an unbuffered FileIO (Popen bufsize=0).
FileIO.read(size) makes one raw read; it does not wait to fill a BufferedReader.
All complete frames supplied by that read retain its immediately captured
timestamp, including frames queued after an earlier frame's logging work.
"""
import time

from tools.splendorust_timing_opt_20261024.common import CandidateStudyWorker

READ_CHUNK_BYTES = 64 * 1024


class BufferedStudyWorker(CandidateStudyWorker):
    """Change only stdout framing; inherit clocks, supervision and cleanup."""

    def _stdout_reader(self):
        pending = bytearray()
        try:
            while True:
                previous_length = len(pending)
                raw = self.process.stdout.read(min(READ_CHUNK_BYTES,
                    self.max_message_bytes + 1 - previous_length))
                received = time.perf_counter_ns()  # Before framing, logging or decoding.
                if not raw:
                    if pending:
                        self._log('stdout.log', bytes(pending[:4096]))
                        self._put(('malformed', received, 'oversized or unterminated message'))
                    else:
                        self._put(('eof', received, None))
                    return
                pending.extend(raw)
                start = 0
                # The retained prefix was already checked for a newline. Scan
                # only newly read bytes; no quadratic rescans of long frames.
                end = pending.find(b'\n', previous_length)
                while end >= 0:
                    frame = bytes(pending[start:end+1])
                    if len(frame) > self.max_message_bytes:
                        self._log('stdout.log', frame[:4096])
                        self._put(('malformed', received, 'oversized or unterminated message'))
                        return
                    self._log('stdout.log', frame)
                    self._put(('line', received, frame))
                    start = end+1
                    end = pending.find(b'\n', start)
                if start:
                    del pending[:start]
                if len(pending) > self.max_message_bytes:
                    self._log('stdout.log', bytes(pending[:4096]))
                    self._put(('malformed', received, 'oversized or unterminated message'))
                    return
        except (OSError, ValueError) as error:
            self._put(('reader_error', time.perf_counter_ns(), str(error)))
