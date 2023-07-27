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

uniform vec4 color;
in vec3 fnormal;
uniform vec3 lightDir;
out vec4 fragColour;

void main()
{
    // Shading styles:
    // 1 - Cartoon: ambient colour, then if dot(fnormal, lightDir) < THRESHOLD, darker colour -> emphasizes edges.
    // 2 - Phong: diffuse, ambient, specular
    // 3 - Translucent: no depth testing, alpha's low.

//    // Phong
//    float F_AMBIENT = 0.4;
//    float F_DIFFUSE = 0.5;
//    float F_SPECULAR = 0.1;
//    float SPEC_POWER = 8.0f;
//    vec3 ambient = F_AMBIENT * color.rgb;
//    vec3 diffuse = dot(fnormal, lightDir) * F_DIFFUSE * color.rgb;
//    vec3 viewDir = normalize(-gl_FragCoord.xyz);
//    vec3 reflDir = reflect(-lightDir, fnormal);
//    float specIntensity = pow(max(dot(viewDir, reflDir), 0.0), SPEC_POWER);
//    vec3 specular = F_SPECULAR * specIntensity * vec3(1.0, 1.0, 1.0);
//    fragColour = vec4(ambient + diffuse + specular, color.a);

    // Cartoon
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