- Fixed reduction hanging without making progress on OpenBSD (#56). The
  interestingness test's stdin is now the test file itself rather than data
  piped from Shrink Ray, avoiding an OpenBSD kqueue deadlock that triggered
  whenever the file was bigger than a pipe buffer and the test exited
  without reading stdin. This also stops copying the whole file to the test
  on every call, on all platforms.
