#vertex
#version 420

layout(location = 0) in vec2 mesh_xy;
layout(location = 1) in vec2 mesh_uv;
layout(location = 2) in float x;
layout(location = 3) in float y;
layout(location = 4) in float uncertainty;
layout(location = 5) in float colour_idx;
layout(location = 6) in float state;
layout (binding = 1) uniform sampler2D lut;

uniform mat4 cameraMatrix;

out vec3 fcolour;
out vec2 fuv;

uniform float quad_pixel_size; // gaussian image quad size in pixels
uniform float quad_uncertainty; // gaussian image uncertainty in nm
uniform float pixel_size; // pixel size in nm
uniform float fixed_sigma;

void main()
{

    gl_Position = cameraMatrix * vec4(mesh_xy * quad_pixel_size * uncertainty / quad_uncertainty + vec2(x, y) / pixel_size, 0.0, 1.0);
    fcolour = texture(lut, vec2(colour_idx, 0)).rgb * quad_uncertainty / uncertainty * state;
    //fcolour = vec3(1.0, 1.0, 1.0) * quad_uncertainty / uncertainty * state;
    fuv = mesh_uv;
}

#fragment
#version 420

layout (binding = 0) uniform sampler2D kernel;
layout (location = 0) out vec4 fragmentColour;


in vec3 fcolour;
in vec2 fuv;

void main()
{

    fragmentColour = vec4(texture(kernel, fuv).r * fcolour, 1.0);
}
