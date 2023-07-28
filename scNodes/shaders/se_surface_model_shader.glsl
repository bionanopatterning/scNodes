#vertex
#version 420

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

uniform mat4 vpMat;

out vec3 fnormal;

void main()
{
    gl_Position = vpMat * vec4(pos, 1.0);
    fnormal = normal;
}

#fragment
#version 420

in vec3 fnormal;
out vec4 fragColour;

uniform vec4 color;
uniform vec3 lightDir;
uniform int style;

void main()
{
    if (style == 0) // Cartoon
    {
        float F_AMBIENT = 0.8;
        float F_DIFFUSE = 0.5;
        vec3 ambient = F_AMBIENT * color.rgb;
        float d = dot(fnormal, lightDir);
        if (d < 0.33)
        {
            ambient *= 0.0;
        }
        vec3 diffuse = max(d, 0.0) * F_DIFFUSE * vec3(1.0f, 1.0f, 1.0f);
        fragColour = vec4(ambient + diffuse, color.a);
    }
    else if (style == 1) // Phong
    {
        float F_AMBIENT = 0.0;
        float F_DIFFUSE = 1.0;
        float F_SPECULAR = 0.0;
        float F_EMISSIVE = 0.4;

        float SPEC_POWER = 8.0f;

        vec3 ambient = F_AMBIENT * color.rgb;

        vec3 diffuse = dot(normalize(fnormal), lightDir) * F_DIFFUSE * color.rgb;

        vec3 viewDir = normalize(gl_FragCoord.xyz);
        vec3 reflDir = reflect(-lightDir, fnormal);
        float specIntensity = pow(max(dot(viewDir, reflDir), 0.0), SPEC_POWER);
        vec3 specular = F_SPECULAR * specIntensity * vec3(1.0, 1.0, 1.0);

        vec3 emissive = dot(normalize(fnormal), lightDir) * F_EMISSIVE * vec3(1.0, 1.0, 1.0);
        fragColour = vec4(ambient + diffuse + specular + emissive, color.a);
    }
    else if (style == 2) // Flat
    {
        fragColour = vec4(color.rgb, color.a);
    }
    else if (style == 3)
    {
        fragColour = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
}