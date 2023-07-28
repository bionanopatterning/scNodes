#compute
#version 430

layout(binding = 0) uniform sampler2D overlay;
layout(binding = 1) uniform sampler2D depth;
layout(binding = 2, rgba32f) writeonly uniform image2D target;
layout(local_size_x = 16, local_size_y=16) in;

uniform mat4 ipMat;
uniform mat4 ivMat;
uniform float far;
uniform float near;
uniform vec2 viewportSize;
uniform float pixelSize;

float world_pos_from_depth(float depth)
{
    float z = 2.0 * depth - 1.0;
    return 2.0 * near * far / (near + far - ((2.0 * depth) - 1.0) * (far - near));
}

void main()
{
    // Get Ray start position.
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(px) / viewportSize;

    // Normalized device coordinates to world coordinates.
    float z = 2.0f * texture(depth, uv).r - 1.0f;
    float x = (2.0f * float(px.x)) / viewportSize.x - 1.0f;
    float y = (2.0f * float(px.y)) / viewportSize.y - 1.0f;
    vec4 ndc = vec4(x, y, z, 1.0);
    vec4 clip_space_vector = ipMat * ndc;
    clip_space_vector /= clip_space_vector.w;

    vec4 pos = ivMat * clip_space_vector;


    // If the depth mask is the clearDepth(1.0f) value, discard ray.
    if (z == 1.0f)
    {
        imageStore(target, px, vec4(1.0f, 0.0f, 0.0f, 1.0f));
    }
    else
    {
        // Find where to start sampling.
        float dl = 1.0f / 13.92f; // TODO: 1.0 / pixelSize;
        float l = 0.0f;

        // Start sampling through volume.

        // When reaching depth > depth mask, end sampling.

        // Write to texture.

        imageStore(target, px, vec4(pos.xyz / 1000.0f, 1.0f));
    }
}
