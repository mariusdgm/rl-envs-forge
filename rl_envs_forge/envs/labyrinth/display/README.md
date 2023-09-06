# Labyrinth Display

This folder contains the code for displaying and rendering the Labyrinth environment.

## Introduction

The Labyrinth Display module provides functionality for visualizing the maze environment and rendering it on a graphical interface using Pygame.

## Contents

- [EnvDisplay](#envdisplay)
- [PlayerDisplayer](#playerdisplayer)

## EnvDisplay

The `EnvDisplay` class is responsible for displaying the maze environment and rendering it on a graphical interface. It provides methods for rendering the maze state and updating the display.

### Rendering the Maze State

Rendering the maze state involves the following steps, as implemented in the `draw_state` method:

1. **Clear Display**: The display surface is cleared to prepare for rendering the new maze state.

2. **Render Maze**: The walls and paths of the maze are rendered onto the display surface.

3. **Render Target**: The corridors of the maze are rendered onto the display surface.

4. **Render Grid Lines**: Display grid lines to help differentiate cells.

4. **Render Entities**: The entities, such as the player and enemies, are rendered onto the display surface.

6. **Update Display**: The display is updated to show the current maze state.

### PlayerDisplayer

The `PlayerDisplayer` class is responsible for rendering the player entity on the maze display. It provides methods for updating the player's position and rendering the player's image on the display.

