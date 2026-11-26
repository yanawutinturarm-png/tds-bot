@echo off
rem Same as "Start TDS Bot.bat", with the Minigunner loadout (config_minigunner.json):
rem   hotbar 1 Soldier, 2 Minigunner, 3 Crook Boss, 4 Commander, 5 Warden (no Farm).
rem   Late in the match every free tower slot gets a level-4 Minigunner.
call "%~dp0Start TDS Bot.bat" --config config_minigunner.json %*
