#vertex
#version 420

layout(location = 0) in vec2 xy;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(xy, 1.0, 1.0);
}

#fragment
#version 420

out vec4 fragmentColour;

uniform vec3 lineColour;

void main()
{
    fragmentColour = vec4(lineColour, 1.0);
}