#version 330

// Input attributes from vertex shader
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

// Output fragment color
out vec4 finalColor;

// Uniforms
uniform sampler2D texture0;

void main() {
    vec4 texelColor = texture(texture0, fragTexCoord) * fragColor;
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));

    float diffuse = max(dot(normal, lightDir), 0.0);
    finalColor = vec4(texelColor.rgb * diffuse, texelColor.a);
}

