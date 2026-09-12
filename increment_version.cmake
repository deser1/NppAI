set(RC_FILE "${CMAKE_CURRENT_LIST_DIR}/src/NppAI_Core.rc")
file(READ ${RC_FILE} RC_CONTENT)

# Wyszukiwanie aktualnej wersji
string(REGEX MATCH "VERSION_DIGITALVALUE ([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)" MATCHED "${RC_CONTENT}")
if(MATCHED)
    set(V1 ${CMAKE_MATCH_1})
    set(V2 ${CMAKE_MATCH_2})
    set(V3 ${CMAKE_MATCH_3})
    set(V4 ${CMAKE_MATCH_4})
    
    # Zwiększenie numeru buildu o 1
    math(EXPR V4 "${V4} + 1")
    
    # Podmiana starych wartości na nowe w pliku
    string(REGEX REPLACE "VERSION_VALUE \"[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+\\\\0\"" "VERSION_VALUE \"${V1}.${V2}.${V3}.${V4}\\\\0\"" RC_CONTENT "${RC_CONTENT}")
    string(REGEX REPLACE "VERSION_DIGITALVALUE [0-9]+, [0-9]+, [0-9]+, [0-9]+" "VERSION_DIGITALVALUE ${V1}, ${V2}, ${V3}, ${V4}" RC_CONTENT "${RC_CONTENT}")
    
    file(WRITE ${RC_FILE} "${RC_CONTENT}")
    message(STATUS "Auto-Versioning: Zaktualizowano wersję wtyczki do ${V1}.${V2}.${V3}.${V4}")
else()
    message(WARNING "Auto-Versioning: Nie znaleziono wersji w pliku .rc")
endif()