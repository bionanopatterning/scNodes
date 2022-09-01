#vertex
#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 UV;
out vec2 fUV;

uniform vec3 translation;
uniform mat4 cameraMatrix;

void main()
{
    gl_Position = cameraMatrix * vec4(position.xyz + translation.xyz, 1.0);
    fUV = UV;
}

#fragment
#version 420

layout (binding = 0) uniform sampler2D image;
layout (binding = 1) uniform sampler2D lut;

out vec4 fragmentColor;

in vec2 fUV;
uniform vec3 contrast_min;
uniform vec3 contrast_max;
uniform int use_lut;

void main()
{
    if (use_lut == 1)
    {
        float pixelValue = (float(texture(image, fUV)).r);
        pixelValue -= contrast_min.r;
        pixelValue /= (contrast_max.r - contrast_min.r);
        pixelValue = clamp(pixelValue, 0.0001, 0.9999);
        vec3 pixelColor = texture(lut, vec2(0, pixelValue)).rgb;
        fragmentColor = vec4(pixelColor, 1.0);
    }
    else
    {
        vec3 pixelValue = vec3(texture(image, fUV));
        pixelValue -= contrast_min;
        pixelValue /= (contrast_max - contrast_min);
        fragmentColor = vec4(pixelValue, 1.0);
    }

}