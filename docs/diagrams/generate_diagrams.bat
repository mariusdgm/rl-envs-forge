@echo off
pushd %~dp0\..
set "ROOT_DIR=%CD%"
pyreverse -o png -p rl_envs_forge rl_envs_forge
move classes_rl_envs_forge.png "%ROOT_DIR%\diagrams\"
move packages_rl_envs_forge.png "%ROOT_DIR%\diagrams\"
echo Diagrams have been generated and moved to the diagrams folder ("%ROOT_DIR%\diagrams\").
popd
@REM pause 
