#vertex
#version 420

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 UV;
out vec2 fUV;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(position, 0.0, 1.0);
    fUV = UV;
}

#fragment
#version 420

layout(binding = 0) uniform usampler2D image;
out vec4 fragmentColor;

in vec2 fUV;

void main()
{
    fragmentColor = vec4(texture(image, fUV));
}