# Baker Node
![baker_nodes](https://user-images.githubusercontent.com/111190478/232863495-25dc2284-66bc-4b09-b56f-94e3061a5c41.png)

Baker Node is a node for Blender's shader editor that can quickly bake its input
to an image, color attribute or sculpt mask.

## Installation
Download the latest baker_node ZIP file from the releases section, then from the
Add-ons section of Blender’s preferences click *Install...* and select the
downloaded .zip file. Enable the add-on labelled *“Node: Baker Node”*.

## Uses
- Quickly create stencil masks or brush masks/textures from shader nodes.
- Use shader nodes to set or modify an object's sculpt mask.
- Improve shader performance by baking the results of complex node setups.
- Easily export texture maps from any part of the shader.

## Features
- Can bake to an image, color attribute or the sculpt mask of the active object.
- Supports baking in the background (Blender 3.3+).
- Baker nodes can be synced so that multiple nodes can be baked with one click.
- Option to automatically create target images or color attributes when baking.

## Usage
**Location:** Shader Editor > Add > Output > Baker

Connect the baker node's input, set the target type and (optionally) select or
create the image/color attribute to bake to. Then press the bake button on the
node.

If no baking target is set then one can be created automatically (if enabled in
the add-on settings).

The baker node's *Baked* socket outputs the value of the target image or color
attribute. Useful for improving a shader's performance by connecting the baker
node after slow sections of the shader.

### Target Types
- **Image (UV)** - Bake to an image using a UV map of the active object.

- **Image (Plane)** - Bake to an image using the co-ordinates of a plane.
Useful for creating brush masks/textures.

- **Color Attribute** - Bake to a color attribute on the active object.

- **Sculpt Mask** - Replace or modify the active object's sculpt mask.

## License
Licensed under the GNU General Public License, version 2.
See [LICENSE.txt](/LICENSE.txt) for details.