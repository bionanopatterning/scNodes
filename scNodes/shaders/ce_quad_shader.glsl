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

uniform float alpha;
uniform vec3 contrastMin;
uniform vec3 contrastMax;
uniform float rgbMode;
uniform float hpix;
uniform float vpix;
uniform float binning;

void main()
{
    vec2 uv = fUV;
    if (binning != 1.0)
    {
        uv.x = floor((uv.x * hpix) / binning) / hpix * binning + (1.0 / hpix) * binning / 2.0;
        uv.y = floor((uv.y * vpix) / binning) / vpix * binning + (1.0 / vpix) * binning / 2.0;
    }
    if (rgbMode == 1.0)
    {
        vec3 rgb = texture(image, uv).rgb;
        float r = (rgb.r - contrastMin.r) / (contrastMax.r - contrastMin.r);
        float g = (rgb.g - contrastMin.g) / (contrastMax.g - contrastMin.g);
        float b = (rgb.b - contrastMin.b) / (contrastMax.b - contrastMin.b);
        fragmentColor = vec4(r, g, b, alpha);
    }
    else
    {
        float pixelValue = texture(image, uv).r;
        float contrastValue = (pixelValue - contrastMin.r) / (contrastMax.r - contrastMin.r);
        vec4 pixelColor = texture(lut, vec2(contrastValue, 0));
        if (pixelColor.a == 0.0)
        {
            discard;
        }
        fragmentColor = vec4(pixelColor.rgb, alpha);
    }
}