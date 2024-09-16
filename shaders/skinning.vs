#version 330

// Input vertex attributes
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;
in vec4 vertexColor;
in ivec4 boneIds;
in vec4 boneWeights;

// Uniforms
uniform mat4 mvp;
uniform mat4 boneTransforms[64]; // Use a fixed size, e.g., 64

// Output attributes
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;

void main() {
    mat4 skinMatrix = boneTransforms[boneIds[0]] * boneWeights[0]
                    + boneTransforms[boneIds[1]] * boneWeights[1]
                    + boneTransforms[boneIds[2]] * boneWeights[2]
                    + boneTransforms[boneIds[3]] * boneWeights[3];

    vec4 position = skinMatrix * vec4(vertexPosition, 1.0);
    gl_Position = mvp * position;

    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = (skinMatrix * vec4(vertexNormal, 0.0)).xyz;
}
