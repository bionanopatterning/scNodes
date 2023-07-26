#vertex
#version 420

layout(location = 0) in vec3 pos;
//layout(location = 1) in vec3 normal;

uniform mat4 vpMat;

out vec3 fnormal;
//out vec3 lightDir;

void main()
{
    gl_Position = vpMat * vec4(1.0 * pos, 1.0);
    //lightDir = vpmat * vec4(0.0, 1.0, 0.0, 1.0);
    fnormal = vec3(1.0, 1.0, 1.0);
}

#fragment
#version 420

uniform vec4 color;
in vec3 fnormal;
const vec3 lightDir = vec3(0.0f, 1.0f, 0.0f);
const float F_AMBIENT = 0.3f;
const float F_DIFFUSE = 0.4f;
const float F_SPECULAR = 0.1f;
const float SPEC_POWER = 4.0f;
out vec4 fragColour;

void main()
{

    vec3 ambient = F_AMBIENT * color.rgb;
    vec3 diffuse = max(dot(fnormal, lightDir), 0.0) * F_DIFFUSE * color.rgb;
    vec3 viewDir = normalize(-gl_FragCoord.xyz);
    vec3 reflDir = reflect(-lightDir, fnormal);
    float specIntensity = pow(max(dot(viewDir, reflDir), 0.0), SPEC_POWER);
    vec3 specular = F_SPECULAR * specIntensity * vec3(1.0, 1.0, 1.0);

    fragColour = vec4(ambient + diffuse + specular, color.a);
}