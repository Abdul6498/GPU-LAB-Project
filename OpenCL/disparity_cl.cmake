set(XIF "???" CACHE STRING "")
set(XOF "???" CACHE STRING "")

# https://stackoverflow.com/a/47801116
file(READ ${XIF} content)
set(delim "for_c++_include")
set(content "R\"${delim}(\n${content})${delim}\"")
file(WRITE ${XOF} "${content}")
