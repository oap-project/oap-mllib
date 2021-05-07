CLANG_FORMAT=$(which clang-format)

$CLANG_FORMAT -style="{ BasedOnStyle: LLVM, \
    UseTab: Never, \
    IndentWidth: 4, \
    TabWidth: 4 \
    }" \
    -dump-config > .clang-format
