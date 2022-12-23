#vertex
#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 UV;
out vec2 fUV;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(position, 1.0);
    fUV = UV;
}

#fragment
#version 420

layout(binding = 0) uniform sampler2D image;
layout(binding = 1) uniform sampler2D lut;

out vec4 fragmentColor;

in vec2 fUV;

uniform vec2 contrastLimits;

void main()
{
    float pixelValue = texture(image, fUV).r;
    float contrastValue = (pixelValue - contrastLimits.x) / (contrastLimits.y - contrastLimits.x);
    vec3 pixelColor = texture(lut, vec2(contrastValue, 0)).rgb;
    fragmentColor = vec4(pixelColor, 1.0);
}