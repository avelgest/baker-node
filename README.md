# Baker Node
![Baker nodes with different target types](https://github.com/avelgest/baker-node/assets/111190478/5e452282-3692-4115-af62-3eb7333c6b15)

Baker Node adds a node to Blender's shader editor that can quickly bake its input
to an image, color attribute or sculpt mask.

## Installation
Download the latest baker_node ZIP file from the releases section, then from the
Add-ons section of Blender’s preferences click *Install...* and select the
downloaded .zip file. Enable the add-on labelled *“Node: Baker Node”*.

## Uses
- Quickly create stencil masks or brush masks/textures from shader nodes.
- Use shader nodes to set or modify an object's sculpt mask.
- Improve shader performance by baking the results of complex node setups.
- Use the outputs of various Cycles-only nodes in Eevee.
- Easily export texture maps from any part of the shader.

## Features
- Can bake to an image, color attribute or the sculpt mask of the active object.
- Supports baking in the background (Blender 3.3+).
- Baker nodes can be synced so that multiple nodes can be baked with one click.
- Option to automatically create target images or color attributes when baking.
- Supports baking both color and alpha.

## Target Types
- **Image (UV)** - Bake to an image using a UV map of the active object.

- **Image (Plane)** - Bake to an image using the co-ordinates of a plane.
Useful for creating brush textures/masks. <br> An automatically updating preview
of the result can be displayed by expanding the *Preview* section on the node.

- **Color Attribute** - Bake to a color attribute on the active object.

- **Sculpt Mask** - Replace or modify the active object's sculpt mask.

## Usage
**Location:** Shader Editor > Add > Output > Baker

Connect the baker node's input, set the target type and (optionally) select or
create the image/color attribute to bake to. Then press the bake button on the
node.

If no baking target is set then one can be created automatically (if enabled in
the add-on settings).

The baker node's *Baked Color* and *Baked Alpha* sockets output values from the
target image or color attribute. They can be used to improving a shader's
performance by inserting the baker node after slow sections of the shader.

Settings such as the number of samples and the object to use when baking can be
set in the node's *Settings* menu accessed by clicking the gear icon on the right
of the node.

## License
Licensed under the GNU General Public License, version 2.
See [LICENSE.txt](/LICENSE.txt) for details.