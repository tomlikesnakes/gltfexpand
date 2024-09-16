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
uniform mat4 model;
uniform mat4 boneTransforms[256]; // Increased from 64 to 256 for more bones if needed

// Output attributes
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
out vec3 fragPosition;

void main() {
    mat4 skinMatrix = mat4(0.0);
    for(int i = 0; i < 4; i++) {
        skinMatrix += boneTransforms[boneIds[i]] * boneWeights[i];
    }

    vec4 worldPosition = model * skinMatrix * vec4(vertexPosition, 1.0);
    gl_Position = mvp * worldPosition;

    fragPosition = worldPosition.xyz;
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = mat3(model * skinMatrix) * vertexNormal;
}
