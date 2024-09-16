// gltf_loader.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

// Define maximum limits
#define MAX_BONES 256
#define MAX_ANIMATIONS 64
#define MAX_MESHES 64
#define MAX_NODES 512
#define MAX_MATERIALS 64
#define MAX_TEXTURES 64

// Enums
typedef enum {
    ATTRIBUTE_POSITION = 0,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEXCOORD,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_TANGENT,
    ATTRIBUTE_BONE_IDS,
    ATTRIBUTE_BONE_WEIGHTS
} AttributeType;

// Data structures
typedef struct {
    Matrix transform;
    Matrix globalTransform;
    int parentIndex;
    int childCount;
    int childIndices[MAX_NODES];
    const char* name;
    bool hasAnimation;
} Node;

typedef struct {
    int nodeIndex;
    const char* name;
} MyBoneInfo;

typedef struct {
    int boneCount;
    MyBoneInfo bones[MAX_BONES];
} Skeleton;

typedef struct {
    int keyframeCount;
    float* times;
    Vector3* translations;
    Quaternion* rotations;
    Vector3* scales;
} AnimationChannel;

typedef struct {
    const char* name;
    int channelCount;
    AnimationChannel channels[MAX_NODES];
    float duration;
} AnimationData;

typedef struct {
    int animationCount;
    AnimationData animations[MAX_ANIMATIONS];
    int currentAnimation;
    bool isPlaying;
    float time;
} AnimationState;

typedef struct {
    Texture2D textures[MAX_TEXTURES];
    int textureCount;
} TextureCache;

typedef struct {
    float* vertices;       // Vertex positions
    float* normals;        // Vertex normals
    float* texcoords;      // Texture coordinates
    unsigned char* colors; // Vertex colors (RGBA)
    float* tangents;       // Vertex tangents
    unsigned short* indices;   // Indices
    unsigned char* boneIds;    // Bone IDs (4 per vertex)
    float* boneWeights;        // Bone weights (4 per vertex)
    int vertexCount;
    int indexCount;
} CustomMesh;

typedef struct {
    unsigned int vaoId;
    unsigned int vboId[8];  // 0:vertices, 1:normals, 2:texcoords, 3:colors, 4:tangents, 5:indices, 6:boneIds, 7:boneWeights
    int vertexCount;
    int indexCount;
        Mesh mesh;  

} CustomMeshGPU;

typedef struct {
    Shader shader;
    Color color;
    Texture2D albedoMap;
    Texture2D normalMap;
    Texture2D metallicRoughnessMap;
    Texture2D emissiveMap;
    Texture2D occlusionMap;
    float metallic;
    float roughness;
    float emissiveFactor[3];
} CustomMaterial;

typedef struct {
    char name[32];
    int parent;
} CustomBoneInfo;

typedef struct {
    Vector3 translation;
    Quaternion rotation;
    Vector3 scale;
} CustomTransform;

typedef struct {
    CustomMesh* meshes;
    CustomMeshGPU* meshesGPU;
    int meshCount;
    CustomMaterial* materials;
    int* meshMaterial;
    int materialCount;
    CustomBoneInfo* bones;
    CustomTransform* bindPose;
    int boneCount;
    Matrix transform;
} CustomModel;

// Function prototypes
cgltf_data* LoadGLTFFile(const char* filename);
void FreeGLTFData(cgltf_data* data);
void LoadNodes(cgltf_data* data, Node* nodes, int* nodeCount);
void BuildNodeHierarchy(Node* nodes, int nodeCount, cgltf_data* data);
void LoadSkeleton(cgltf_data* data, Skeleton* skeleton, Node* nodes, int nodeCount);
void LoadAnimations(cgltf_data* data, AnimationState* animationState, Node* nodes, int nodeCount);
CustomModel LoadCustomModelFromGLTF(cgltf_data* data, Skeleton* skeleton, TextureCache* textureCache, Node* nodes, int nodeCount, MyBoneInfo** outMyBoneInfos, Matrix** outInverseBindPose);
void UpdateAnimation(AnimationState* animationState, Node* nodes, int nodeCount);
void ApplyNodeTransforms(Node* nodes, int nodeCount, Skeleton* skeleton, MyBoneInfo* myBoneInfos, Matrix* inverseBindPose, Matrix* boneTransforms);
void UpdateGlobalTransforms(Node* nodes, int nodeIndex);
void FreeAnimationData(AnimationState* animationState, int nodeCount);
void LoadCustomMaterials(cgltf_data* data, CustomModel* model, TextureCache* textureCache);
Texture2D LoadTextureFromImageUri(const char* basePath, const char* uri);
void PrintNodeHierarchy(Node* nodes, int nodeIndex, int depth);
void PrintBoneMatrices(Matrix* boneTransforms, Skeleton* skeleton);
void PrintDebugInfo(AnimationState* animationState, Skeleton* skeleton, Node* nodes, int nodeCount, Matrix* boneTransforms);
void PrintMeshData(CustomMesh* mesh);
void SetShaderValueMatrixArray(Shader shader, int locIndex, const Matrix* matrices, int count);
Vector3 MatrixToTranslation(Matrix m);
Vector3 MatrixToScale(Matrix m);
Quaternion QuaternionFromMatrix(Matrix m);
void DrawSkeleton(CustomModel model, Matrix* boneTransforms);
Matrix TransformToMatrix(CustomTransform t);
void UnloadCustomModel(CustomModel model);
void DrawCustomModel(CustomModel model, Matrix transform, Matrix* boneTransforms, Camera camera);
void UploadCustomMesh(CustomMesh* mesh, CustomMeshGPU* meshGPU);

// Shader code
const char* vertexShaderSkinning =
    "#version 330\n"
    "in vec3 vertexPosition;\n"
    "in vec3 vertexNormal;\n"
    "in vec2 vertexTexCoord;\n"
    "in vec4 vertexColor;\n"
    "in vec4 vertexTangent;\n"
    "in ivec4 vertexBoneIds;\n"
    "in vec4 vertexBoneWeights;\n"
    "uniform mat4 mvp;\n"
    "uniform mat4 boneTransforms[256];\n"
    "out vec3 fragPosition;\n"
    "out vec2 fragTexCoord;\n"
    "out vec3 fragNormal;\n"
    "out vec4 fragColor;\n"
    "out mat3 TBN;\n"
    "void main()\n"
    "{\n"
    "    mat4 skinMatrix = mat4(0.0);\n"
    "    for(int i = 0; i < 4; i++)\n"
    "    {\n"
    "        skinMatrix += boneTransforms[vertexBoneIds[i]] * vertexBoneWeights[i];\n"
    "    }\n"
    "    vec4 skinnedPosition = skinMatrix * vec4(vertexPosition, 1.0);\n"
    "    vec3 skinnedNormal = mat3(skinMatrix) * vertexNormal;\n"
    "    vec3 skinnedTangent = mat3(skinMatrix) * vertexTangent.xyz;\n"
    "    gl_Position = mvp * skinnedPosition;\n"
    "    fragPosition = skinnedPosition.xyz;\n"
    "    fragTexCoord = vertexTexCoord;\n"
    "    fragNormal = normalize(skinnedNormal);\n"
    "    fragColor = vertexColor;\n"
    "    vec3 N = normalize(skinnedNormal);\n"
    "    vec3 T = normalize(skinnedTangent);\n"
    "    vec3 B = normalize(cross(N, T)) * vertexTangent.w;\n"
    "    TBN = mat3(T, B, N);\n"
    "}\n";

const char* fragmentShaderPBR =
    "#version 330\n"
    "in vec3 fragPosition;\n"
    "in vec2 fragTexCoord;\n"
    "in vec3 fragNormal;\n"
    "in vec4 fragColor;\n"
    "in mat3 TBN;\n"
    "uniform sampler2D albedoMap;\n"
    "uniform sampler2D normalMap;\n"
    "uniform sampler2D metallicRoughnessMap;\n"
    "uniform sampler2D emissiveMap;\n"
    "uniform sampler2D occlusionMap;\n"
    "uniform vec4 baseColor;\n"
    "uniform float metallic;\n"
    "uniform float roughness;\n"
    "uniform vec3 emissiveFactor;\n"
    "uniform vec3 cameraPos;\n"
    "out vec4 finalColor;\n"
    "const float PI = 3.14159265359;\n"
    "vec3 fresnelSchlick(float cosTheta, vec3 F0)\n"
    "{\n"
    "    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);\n"
    "}\n"
    "float DistributionGGX(vec3 N, vec3 H, float roughness)\n"
    "{\n"
    "    float a = roughness*roughness;\n"
    "    float a2 = a*a;\n"
    "    float NdotH = max(dot(N, H), 0.0);\n"
    "    float NdotH2 = NdotH*NdotH;\n"
    "    float num = a2;\n"
    "    float denom = (NdotH2 * (a2 - 1.0) + 1.0);\n"
    "    denom = PI * denom * denom;\n"
    "    return num / denom;\n"
    "}\n"
    "float GeometrySchlickGGX(float NdotV, float roughness)\n"
    "{\n"
    "    float r = (roughness + 1.0);\n"
    "    float k = (r*r) / 8.0;\n"
    "    float num = NdotV;\n"
    "    float denom = NdotV * (1.0 - k) + k;\n"
    "    return num / denom;\n"
    "}\n"
    "float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)\n"
    "{\n"
    "    float NdotV = max(dot(N, V), 0.0);\n"
    "    float NdotL = max(dot(N, L), 0.0);\n"
    "    float ggx2 = GeometrySchlickGGX(NdotV, roughness);\n"
    "    float ggx1 = GeometrySchlickGGX(NdotL, roughness);\n"
    "    return ggx1 * ggx2;\n"
    "}\n"
    "void main()\n"
    "{\n"
    "    vec3 albedo = texture(albedoMap, fragTexCoord).rgb * baseColor.rgb;\n"
    "    float alpha = texture(albedoMap, fragTexCoord).a * baseColor.a;\n"
    "    vec3 normal = texture(normalMap, fragTexCoord).rgb;\n"
    "    normal = normalize(normal * 2.0 - 1.0);\n"
    "    normal = normalize(TBN * normal);\n"
    "    vec2 metallicRoughness = texture(metallicRoughnessMap, fragTexCoord).bg;\n"
    "    float metalli = metallicRoughness.x * metallic;\n"
    "    float roughnes = metallicRoughness.y * roughness;\n"
    "    vec3 emissive = texture(emissiveMap, fragTexCoord).rgb * emissiveFactor;\n"
    "    float ao = texture(occlusionMap, fragTexCoord).r;\n"
    "    vec3 N = normalize(normal);\n"
    "    vec3 V = normalize(cameraPos - fragPosition);\n"
    "    vec3 F0 = vec3(0.04);\n"
    "    F0 = mix(F0, albedo, metalli);\n"
    "    vec3 Lo = vec3(0.0);\n"
    "    vec3 L = normalize(vec3(1.0, 1.0, 0.0));\n"
    "    vec3 H = normalize(V + L);\n"
    "    float NdotL = max(dot(N, L), 0.0);\n"
    "    vec3 radiance = vec3(1.0);\n"
    "    float NDF = DistributionGGX(N, H, roughnes);\n"
    "    float G = GeometrySmith(N, V, L, roughnes);\n"
    "    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);\n"
    "    vec3 numerator = NDF * G * F;\n"
    "    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);\n"
    "    vec3 specular = numerator / max(denominator, 0.001);\n"
    "    vec3 kS = F;\n"
    "    vec3 kD = vec3(1.0) - kS;\n"
    "    kD *= 1.0 - metalli;\n"
    "    Lo += (kD * albedo / PI + specular) * radiance * NdotL;\n"
    "    vec3 ambient = vec3(0.03) * albedo * ao;\n"
    "    vec3 color = ambient + Lo + emissive;\n"
    "    color = color / (color + vec3(1.0));\n"
    "    color = pow(color, vec3(1.0/2.2));\n"
    "    finalColor = vec4(color, alpha);\n"
    "}\n";

const char *basicVertexShader =
    "#version 330\n"
    "in vec3 vertexPosition;\n"
    "in vec2 vertexTexCoord;\n"
    "in vec3 vertexNormal;\n"
    "in vec4 vertexColor;\n"
    "uniform mat4 mvp;\n"
    "out vec2 fragTexCoord;\n"
    "out vec4 fragColor;\n"
    "void main()\n"
    "{\n"
    "    fragTexCoord = vertexTexCoord;\n"
    "    fragColor = vertexColor;\n"
    "    gl_Position = mvp * vec4(vertexPosition, 1.0);\n"
    "}\n";

const char *basicFragmentShader =
    "#version 330\n"
    "in vec2 fragTexCoord;\n"
    "in vec4 fragColor;\n"
    "out vec4 finalColor;\n"
    "uniform sampler2D texture0;\n"
    "uniform vec4 colDiffuse;\n"
    "void main()\n"
    "{\n"
    "    vec4 texelColor = texture(texture0, fragTexCoord);\n"
    "    finalColor = texelColor * colDiffuse * fragColor;\n"
    "}\n";

Matrix MatrixFromFloatArray(float* arr) {
    Matrix result = { 0 };
    result.m0 = arr[0];  result.m4 = arr[4];  result.m8 = arr[8];   result.m12 = arr[12];
    result.m1 = arr[1];  result.m5 = arr[5];  result.m9 = arr[9];   result.m13 = arr[13];
    result.m2 = arr[2];  result.m6 = arr[6];  result.m10 = arr[10]; result.m14 = arr[14];
    result.m3 = arr[3];  result.m7 = arr[7];  result.m11 = arr[11]; result.m15 = arr[15];
    return result;
}


// Function implementations
cgltf_data* LoadGLTFFile(const char* filename) {
    cgltf_options options = {0};
    cgltf_data* data = NULL;

    cgltf_result result = cgltf_parse_file(&options, filename, &data);
    if (result != cgltf_result_success) {
        printf("Failed to parse glTF file: %s\n", filename);
        return NULL;
    }

    result = cgltf_load_buffers(&options, data, filename);
    if (result != cgltf_result_success) {
        printf("Failed to load buffers for glTF file: %s\n", filename);
        cgltf_free(data);
        return NULL;
    }

    result = cgltf_validate(data);
    if (result != cgltf_result_success) {
        printf("Validation failed for glTF file: %s\n", filename);
        cgltf_free(data);
        return NULL;
    }

    printf("Successfully loaded glTF file: %s\n", filename);
    return data;
}

void FreeGLTFData(cgltf_data* data) {
    cgltf_free(data);
}

void LoadNodes(cgltf_data* data, Node* nodes, int* nodeCount) {
    for (cgltf_size i = 0; i < data->nodes_count; ++i) {
        cgltf_node* cgltfNode = &data->nodes[i];
        nodes[*nodeCount].name = cgltfNode->name ? cgltfNode->name : "Unnamed";
        nodes[*nodeCount].parentIndex = -1;
        nodes[*nodeCount].childCount = (int)cgltfNode->children_count;
        nodes[*nodeCount].hasAnimation = false;

        // Copy local transformation
        if (cgltfNode->has_matrix) {
            nodes[*nodeCount].transform = MatrixFromFloatArray((float*)cgltfNode->matrix);
        } else {
            nodes[*nodeCount].transform = MatrixIdentity();
            if (cgltfNode->has_translation) {
                nodes[*nodeCount].transform = MatrixMultiply(nodes[*nodeCount].transform,
                         MatrixTranslate(cgltfNode->translation[0],
               cgltfNode->translation[1],
               cgltfNode->translation[2]));
            }
            if (cgltfNode->has_rotation) {
                Quaternion q = {cgltfNode->rotation[0], cgltfNode->rotation[1],
                    cgltfNode->rotation[2], cgltfNode->rotation[3]};
                nodes[*nodeCount].transform = MatrixMultiply(nodes[*nodeCount].transform,
                         QuaternionToMatrix(q));
            }
            if (cgltfNode->has_scale) {
                nodes[*nodeCount].transform = MatrixMultiply(nodes[*nodeCount].transform,
                         MatrixScale(cgltfNode->scale[0],
           cgltfNode->scale[1],
           cgltfNode->scale[2]));
            }
        }

        nodes[*nodeCount].globalTransform = nodes[*nodeCount].transform;

        (*nodeCount)++;
    }
}

void BuildNodeHierarchy(Node* nodes, int nodeCount, cgltf_data* data) {
    for (int i = 0; i < nodeCount; ++i) {
        cgltf_node* cgltfNode = &data->nodes[i];

        // Set parent index
        if (cgltfNode->parent) {
            // Find parent index
            for (int j = 0; j < nodeCount; ++j) {
                if (&data->nodes[j] == cgltfNode->parent) {
                    nodes[i].parentIndex = j;
                    break;
                }
            }
        }

        // Set child indices
        for (cgltf_size k = 0; k < cgltfNode->children_count; ++k) {
            cgltf_node* childNode = cgltfNode->children[k];
            // Find child index
            for (int j = 0; j < nodeCount; ++j) {
                if (&data->nodes[j] == childNode) {
                    nodes[i].childIndices[k] = j;
                    break;
                }
            }
        }
    }
}

void LoadSkeleton(cgltf_data* data, Skeleton* skeleton, Node* nodes, int nodeCount) {
    if (data->skins_count == 0) {
        printf("No skeleton (skins) found in glTF file.\n");
        return;
    }

    cgltf_skin* skin = &data->skins[0];  // Assuming first skin
    skeleton->boneCount = (int)skin->joints_count;

    printf("\n--- Loading Skeleton with %d bones ---\n", skeleton->boneCount);

    for (int i = 0; i < skeleton->boneCount; ++i) {
        cgltf_node* jointNode = skin->joints[i];
        // Find the index of the jointNode in our nodes array
        int nodeIndex = -1;
        for (int j = 0; j < nodeCount; ++j) {
            if (&data->nodes[j] == jointNode) {
                nodeIndex = j;
                break;
            }
        }
        skeleton->bones[i].nodeIndex = nodeIndex;
        skeleton->bones[i].name = nodes[nodeIndex].name;

        printf("Bone %d: %s (Node Index: %d)\n", i, skeleton->bones[i].name, nodeIndex);
    }
}

void LoadAnimations(cgltf_data* data, AnimationState* animationState, Node* nodes, int nodeCount) {
    animationState->animationCount = (int)data->animations_count;
    printf("\n--- Loading %d animations ---\n", animationState->animationCount);

    for (int a = 0; a < animationState->animationCount; ++a) {
        cgltf_animation* animation = &data->animations[a];
        AnimationData* animData = &animationState->animations[a];
        animData->name = animation->name ? animation->name : "Unnamed Animation";
        animData->channelCount = (int)animation->channels_count;
        animData->duration = 0.0f;

        printf("Animation %d: %s\n", a, animData->name);

        // Initialize channels
        for (int i = 0; i < nodeCount; ++i) {
            animData->channels[i].keyframeCount = 0;
            animData->channels[i].times = NULL;
            animData->channels[i].translations = NULL;
            animData->channels[i].rotations = NULL;
            animData->channels[i].scales = NULL;
        }

        for (int i = 0; i < animData->channelCount; ++i) {
            cgltf_animation_channel* channel = &animation->channels[i];
            cgltf_animation_sampler* sampler = channel->sampler;

            // Find the index of the target node
            int nodeIndex = -1;
            for (int j = 0; j < nodeCount; ++j) {
                if (&data->nodes[j] == channel->target_node) {
                    nodeIndex = j;
                    nodes[j].hasAnimation = true;
                    break;
                }
            }

            if (nodeIndex == -1) {
                printf("Warning: Animation channel target node not found.\n");
                continue;
            }

            AnimationChannel* animChannel = &animData->channels[nodeIndex];

            // Load input times
            cgltf_accessor* inputAccessor = sampler->input;
            size_t keyframeCount = inputAccessor->count;
            animChannel->keyframeCount = (int)keyframeCount;
            animChannel->times = malloc(sizeof(float) * keyframeCount);
            for (size_t k = 0; k < keyframeCount; ++k) {
                cgltf_accessor_read_float(inputAccessor, k, &animChannel->times[k], 1);
                if (animChannel->times[k] > animData->duration) {
                    animData->duration = animChannel->times[k];
                }
            }

            // Load output values
            cgltf_accessor* outputAccessor = sampler->output;
            if (channel->target_path == cgltf_animation_path_type_translation) {
                animChannel->translations = malloc(sizeof(Vector3) * keyframeCount);
                for (size_t k = 0; k < keyframeCount; ++k) {
                    float value[3];
                    cgltf_accessor_read_float(outputAccessor, k, value, 3);
                    animChannel->translations[k] = (Vector3){value[0], value[1], value[2]};
                }
            } else if (channel->target_path == cgltf_animation_path_type_rotation) {
                animChannel->rotations = malloc(sizeof(Quaternion) * keyframeCount);
                for (size_t k = 0; k < keyframeCount; ++k) {
                    float value[4];
                    cgltf_accessor_read_float(outputAccessor, k, value, 4);
                    animChannel->rotations[k] = (Quaternion){value[0], value[1], value[2], value[3]};
                }
            } else if (channel->target_path == cgltf_animation_path_type_scale) {
                animChannel->scales = malloc(sizeof(Vector3) * keyframeCount);
                for (size_t k = 0; k < keyframeCount; ++k) {
                    float value[3];
                    cgltf_accessor_read_float(outputAccessor, k, value, 3);
                    animChannel->scales[k] = (Vector3){value[0], value[1], value[2]};
                }
            }
        }
        printf("Animation '%s' duration: %f seconds\n", animData->name, animData->duration);
    }
}

CustomModel LoadCustomModelFromGLTF(cgltf_data* data, Skeleton* skeleton, TextureCache* textureCache, Node* nodes, int nodeCount, MyBoneInfo** outMyBoneInfos, Matrix** outInverseBindPose) {
    printf("\n--- Loading meshes and materials from glTF file ---\n");
    CustomModel model = {0};

    // Count total number of primitives
    int totalPrimitives = 0;
    for (cgltf_size m = 0; m < data->meshes_count; ++m) {
        cgltf_mesh* cgltfMesh = &data->meshes[m];
        totalPrimitives += (int)cgltfMesh->primitives_count;
    }

    model.meshCount = totalPrimitives;
    printf("Total primitives: %d\n", totalPrimitives);

    model.meshes = (CustomMesh*)calloc(model.meshCount, sizeof(CustomMesh));
    model.meshesGPU = (CustomMeshGPU*)calloc(model.meshCount, sizeof(CustomMeshGPU));
    model.meshMaterial = (int*)calloc(model.meshCount, sizeof(int));

    if (!model.meshes || !model.meshesGPU || !model.meshMaterial) {
        printf("Error: Failed to allocate memory for meshes\n");
        return model;
    }

    int meshIndex = 0;

    // Load meshes
    for (cgltf_size m = 0; m < data->meshes_count; ++m) {
        cgltf_mesh* cgltfMesh = &data->meshes[m];

        for (cgltf_size p = 0; p < cgltfMesh->primitives_count; ++p) {
            cgltf_primitive* primitive = &cgltfMesh->primitives[p];
            CustomMesh* mesh = &model.meshes[meshIndex];

            printf("Loading mesh %d\n", meshIndex);

            // Load indices
            cgltf_accessor* indexAccessor = primitive->indices;
            if (indexAccessor) {
                mesh->indexCount = (int)indexAccessor->count;
                mesh->indices = (unsigned short*)malloc(sizeof(unsigned short) * mesh->indexCount);
                for (size_t i = 0; i < indexAccessor->count; ++i) {
                    cgltf_uint tempIndex;
                    cgltf_accessor_read_uint(indexAccessor, i, &tempIndex, 1);
                    mesh->indices[i] = (unsigned short)tempIndex;
                }
                printf("  Loaded %d indices\n", mesh->indexCount);
            } else {
                printf("Warning: Primitive without indices is not supported.\n");
                continue;
            }

            // Load attributes
            for (cgltf_size a = 0; a < primitive->attributes_count; ++a) {
                cgltf_attribute* attribute = &primitive->attributes[a];
                cgltf_accessor* accessor = attribute->data;

                if (attribute->type == cgltf_attribute_type_position) {
                    mesh->vertexCount = (int)accessor->count;
                    mesh->vertices = (float*)malloc(sizeof(float) * 3 * mesh->vertexCount);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_accessor_read_float(accessor, i, &mesh->vertices[i * 3], 3);
                    }
                    printf("  Loaded %d vertices\n", mesh->vertexCount);
                } else if (attribute->type == cgltf_attribute_type_normal) {
                    mesh->normals = (float*)malloc(sizeof(float) * 3 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_accessor_read_float(accessor, i, &mesh->normals[i * 3], 3);
                    }
                    printf("  Loaded normals\n");
                } else if (attribute->type == cgltf_attribute_type_texcoord) {
                    mesh->texcoords = (float*)malloc(sizeof(float) * 2 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_accessor_read_float(accessor, i, &mesh->texcoords[i * 2], 2);
                    }
                    printf("  Loaded texcoords\n");
                } else if (attribute->type == cgltf_attribute_type_texcoord) {
                    mesh->texcoords = (float*)malloc(sizeof(float) * 2 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_accessor_read_float(accessor, i, &mesh->texcoords[i * 2], 2);
                    }
                } else if (attribute->type == cgltf_attribute_type_color) {
                    mesh->colors = (unsigned char*)malloc(sizeof(unsigned char) * 4 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        float color[4];
                        cgltf_accessor_read_float(accessor, i, color, 4);
                        for (int j = 0; j < 4; ++j) {
                            mesh->colors[i * 4 + j] = (unsigned char)(color[j] * 255.0f);
                        }
                    }
                } else if (attribute->type == cgltf_attribute_type_tangent) {
                    mesh->tangents = (float*)malloc(sizeof(float) * 4 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_accessor_read_float(accessor, i, &mesh->tangents[i * 4], 4);
                    }
                } else if (attribute->type == cgltf_attribute_type_joints) {
                    mesh->boneIds = (unsigned char*)malloc(sizeof(unsigned char) * 4 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_uint joints[4];
                        cgltf_accessor_read_uint(accessor, i, joints, 4);
                        for (int j = 0; j < 4; ++j) {
                            mesh->boneIds[i * 4 + j] = (unsigned char)joints[j];
                        }
                    }
                } else if (attribute->type == cgltf_attribute_type_weights) {
                    mesh->boneWeights = (float*)malloc(sizeof(float) * 4 * accessor->count);
                    for (size_t i = 0; i < accessor->count; ++i) {
                        cgltf_accessor_read_float(accessor, i, &mesh->boneWeights[i * 4], 4);
                    }
                }
            }
            // Assign material
            if (primitive->material) {
                // Find material index
                int materialIndex = -1;
                for (int m = 0; m < model.materialCount; ++m) {
                    if (&data->materials[m] == primitive->material) {
                        materialIndex = m;
                        break;
                    }
                }
                if (materialIndex == -1) {
                    printf("Warning: Material not found for mesh %d. Using default material.\n", meshIndex);
                    materialIndex = 0;
                }
                model.meshMaterial[meshIndex] = materialIndex;
            } else {
                model.meshMaterial[meshIndex] = 0; // Default material
            }

            // Upload mesh to GPU
            UploadCustomMesh(&model.meshes[meshIndex], &model.meshesGPU[meshIndex]);

            printf("Finished loading mesh %d\n", meshIndex);
            meshIndex++;
        }
    }

    // Load materials
    LoadCustomMaterials(data, &model, textureCache);
 
    printf("Initializing bones\n");
    // Initialize bones
    model.boneCount = skeleton->boneCount;
    model.bones = (CustomBoneInfo*)malloc(sizeof(CustomBoneInfo) * model.boneCount);
    model.bindPose = (CustomTransform*)malloc(sizeof(CustomTransform) * model.boneCount);

    if (!model.bones || !model.bindPose) {
        printf("Failed to allocate memory for bones or bind pose\n");
        return model;
    }

    printf("Allocating MyBoneInfo array\n");
    // Allocate MyBoneInfo array
    *outMyBoneInfos = (MyBoneInfo*)malloc(sizeof(MyBoneInfo) * model.boneCount);
    MyBoneInfo* myBoneInfos = *outMyBoneInfos;

    if (!myBoneInfos) {
        printf("Failed to allocate memory for myBoneInfos\n");
        return model;
    }

    printf("Allocating inverse bind pose array\n");
    // Allocate inverse bind pose array
    Matrix* inverseBindPose = (Matrix*)malloc(sizeof(Matrix) * model.boneCount);

    if (!inverseBindPose) {
        printf("Failed to allocate memory for inverseBindPose\n");
        return model;
    }

    printf("Updating global transforms for nodes in bind pose\n");
    // Update global transforms for nodes in bind pose
    for (int i = 0; i < nodeCount; ++i) {
        UpdateGlobalTransforms(nodes, i);
    }

    printf("Processing bones\n");
    for (int i = 0; i < model.boneCount; ++i) {
        int nodeIndex = skeleton->bones[i].nodeIndex;

        printf("Processing bone %d (Node Index: %d)\n", i, nodeIndex);

        // Initialize CustomBoneInfo
        strncpy(model.bones[i].name, skeleton->bones[i].name, sizeof(model.bones[i].name) - 1);
        model.bones[i].name[sizeof(model.bones[i].name) - 1] = '\0'; // Ensure null-termination
        model.bones[i].parent = nodes[nodeIndex].parentIndex;

        // Initialize MyBoneInfo
        myBoneInfos[i].nodeIndex = nodeIndex;
        myBoneInfos[i].name = skeleton->bones[i].name;

        // Capture bind pose transforms
        Matrix globalTransform = nodes[nodeIndex].globalTransform;

        // Store the bind pose
        model.bindPose[i].translation = MatrixToTranslation(globalTransform);
        model.bindPose[i].rotation = QuaternionFromMatrix(globalTransform);
        model.bindPose[i].scale = MatrixToScale(globalTransform);

        // Calculate inverse bind pose
        Matrix bindPoseMatrix = MatrixMultiply(
            MatrixMultiply(
                QuaternionToMatrix(model.bindPose[i].rotation),
                MatrixTranslate(model.bindPose[i].translation.x, model.bindPose[i].translation.y, model.bindPose[i].translation.z)
            ),
            MatrixScale(model.bindPose[i].scale.x, model.bindPose[i].scale.y, model.bindPose[i].scale.z)
        );
        inverseBindPose[i] = MatrixInvert(bindPoseMatrix);
    }

    printf("Setting output inverse bind pose\n");
    // Output inverse bind pose
    *outInverseBindPose = inverseBindPose;

    model.transform = MatrixIdentity();

    printf("CustomModel loading completed\n");
    return model;
}
  
void UpdateAnimation(AnimationState* animationState, Node* nodes, int nodeCount) {
    if (animationState->animationCount == 0) return;

    AnimationData* animData = &animationState->animations[animationState->currentAnimation];
    float time = animationState->time;

    for (int i = 0; i < nodeCount; ++i) {
        if (!nodes[i].hasAnimation) continue;
        AnimationChannel* channel = &animData->channels[i];
        if (channel->keyframeCount == 0) continue;

        // Find the current keyframe
        int frame = 0;
        while (frame < channel->keyframeCount - 1 && channel->times[frame + 1] < time) {
            frame++;
        }

        // Calculate interpolation factor
        float t = 0.0f;
        if (frame < channel->keyframeCount - 1) {
            float startTime = channel->times[frame];
            float endTime = channel->times[frame + 1];
            t = (time - startTime) / (endTime - startTime);
        }

        // Interpolate transformation
        Matrix transform = MatrixIdentity();
        if (channel->translations) {
            Vector3 start = channel->translations[frame];
            Vector3 end = channel->translations[frame + 1];
            Vector3 translation = Vector3Lerp(start, end, t);

            transform = MatrixMultiply(transform, MatrixTranslate(translation.x, translation.y, translation.z));
        }
        if (channel->rotations) {
            Quaternion start = channel->rotations[frame];
            Quaternion end = channel->rotations[frame + 1];
            Quaternion rotation = QuaternionSlerp(start, end, t);

            transform = MatrixMultiply(transform, QuaternionToMatrix(rotation));
        }
        if (channel->scales) {
            Vector3 start = channel->scales[frame];
            Vector3 end = channel->scales[frame + 1];
            Vector3 scale = Vector3Lerp(start, end, t);

            transform = MatrixMultiply(transform, MatrixScale(scale.x, scale.y, scale.z));
        }

        nodes[i].transform = transform;
    }

    // Update global transforms
    for (int i = 0; i < nodeCount; ++i) {
        UpdateGlobalTransforms(nodes, i);
    }
}

void ApplyNodeTransforms(Node* nodes, int nodeCount, Skeleton* skeleton, MyBoneInfo* myBoneInfos, Matrix* inverseBindPose, Matrix* boneTransforms) {
    for (int i = 0; i < skeleton->boneCount; ++i) {
        int nodeIndex = myBoneInfos[i].nodeIndex;
        Matrix globalTransform = nodes[nodeIndex].globalTransform;

        // Compute the final bone transform
        boneTransforms[i] = MatrixMultiply(inverseBindPose[i], globalTransform);
    }
}

void UpdateGlobalTransforms(Node* nodes, int nodeIndex) {
    Node* node = &nodes[nodeIndex];
    if (node->parentIndex == -1) {
        node->globalTransform = node->transform;
    } else {
        node->globalTransform = MatrixMultiply(nodes[node->parentIndex].globalTransform, node->transform);
    }
}

void FreeAnimationData(AnimationState* animationState, int nodeCount) {
    for (int a = 0; a < animationState->animationCount; ++a) {
        AnimationData* animData = &animationState->animations[a];
        for (int i = 0; i < nodeCount; ++i) {
            AnimationChannel* channel = &animData->channels[i];
            if (channel->times) free(channel->times);
            if (channel->translations) free(channel->translations);
            if (channel->rotations) free(channel->rotations);
            if (channel->scales) free(channel->scales);
        }
    }
}

void LoadCustomMaterials(cgltf_data* data, CustomModel* model, TextureCache* textureCache) {
    model->materialCount = (int)data->materials_count;
    if (model->materialCount == 0) {
        printf("No materials found in glTF file. Creating a default material.\n");
        model->materialCount = 1;
        model->materials = (CustomMaterial*)calloc(1, sizeof(CustomMaterial));
        model->materials[0].shader = LoadShaderFromMemory(vertexShaderSkinning, fragmentShaderPBR);
        model->materials[0].color = WHITE;
        model->materials[0].metallic = 0.0f;
        model->materials[0].roughness = 0.5f;
        return;
    }

    model->materials = (CustomMaterial*)calloc(model->materialCount, sizeof(CustomMaterial));
    printf("Allocating %d materials\n", model->materialCount);

    // Base path for texture files
    const char* basePath = GetDirectoryPath(GetWorkingDirectory());

    for (int i = 0; i < model->materialCount; ++i) {
        cgltf_material* cgltfMat = &data->materials[i];
        CustomMaterial* mat = &model->materials[i];

        printf("Loading material %d\n", i);

        // Load shader
        mat->shader = LoadShaderFromMemory(vertexShaderSkinning, fragmentShaderPBR);

        // Set material properties
        if (cgltfMat->has_pbr_metallic_roughness) {
            cgltf_pbr_metallic_roughness* pbr = &cgltfMat->pbr_metallic_roughness;

            // Base color
            mat->color = (Color){
                (unsigned char)(pbr->base_color_factor[0] * 255),
                (unsigned char)(pbr->base_color_factor[1] * 255),
                (unsigned char)(pbr->base_color_factor[2] * 255),
                (unsigned char)(pbr->base_color_factor[3] * 255)
            };

            // Base color texture
            if (pbr->base_color_texture.texture && pbr->base_color_texture.texture->image) {
                const char* uri = pbr->base_color_texture.texture->image->uri;
                if (uri) {
                    mat->albedoMap = LoadTextureFromImageUri(basePath, uri);
                }
            }

            // Metallic and roughness
            mat->metallic = pbr->metallic_factor;
            mat->roughness = pbr->roughness_factor;

            // Metallic-Roughness texture
            if (pbr->metallic_roughness_texture.texture && pbr->metallic_roughness_texture.texture->image) {
                const char* uri = pbr->metallic_roughness_texture.texture->image->uri;
                if (uri) {
                    mat->metallicRoughnessMap = LoadTextureFromImageUri(basePath, uri);
                }
            }
        }

        // Normal map
        if (cgltfMat->normal_texture.texture && cgltfMat->normal_texture.texture->image) {
            const char* uri = cgltfMat->normal_texture.texture->image->uri;
            if (uri) {
                mat->normalMap = LoadTextureFromImageUri(basePath, uri);
            }
        }

        // Occlusion map
        if (cgltfMat->occlusion_texture.texture && cgltfMat->occlusion_texture.texture->image) {
            const char* uri = cgltfMat->occlusion_texture.texture->image->uri;
            if (uri) {
                mat->occlusionMap = LoadTextureFromImageUri(basePath, uri);
            }
        }

        // Emissive map and factor
        if (cgltfMat->emissive_texture.texture && cgltfMat->emissive_texture.texture->image) {
            const char* uri = cgltfMat->emissive_texture.texture->image->uri;
            if (uri) {
                mat->emissiveMap = LoadTextureFromImageUri(basePath, uri);
            }
        }
                printf("Finished loading material %d\n", i);

        memcpy(mat->emissiveFactor, cgltfMat->emissive_factor, sizeof(float) * 3);
    }
}

Texture2D LoadTextureFromImageUri(const char* basePath, const char* uri) {
    char fullPath[512];
    snprintf(fullPath, sizeof(fullPath), "%s/%s", basePath, uri);

    printf("Loading texture: %s\n", fullPath);
    Image image = LoadImage(fullPath);
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);
    return texture;
}

void PrintNodeHierarchy(Node* nodes, int nodeIndex, int depth) {
    for (int i = 0; i < depth; ++i) printf("  ");
    printf("Node %d: %s\n", nodeIndex, nodes[nodeIndex].name);
    for (int i = 0; i < nodes[nodeIndex].childCount; ++i) {
        PrintNodeHierarchy(nodes, nodes[nodeIndex].childIndices[i], depth + 1);
    }
}

void PrintBoneMatrices(Matrix* boneTransforms, Skeleton* skeleton) {
    printf("\n--- Bone Matrices ---\n");
    for (int i = 0; i < skeleton->boneCount; ++i) {
        printf("Bone %d: %s\n", i, skeleton->bones[i].name);
        Matrix m = boneTransforms[i];
        printf("Matrix:\n");
        printf("%f %f %f %f\n", m.m0, m.m4, m.m8, m.m12);
        printf("%f %f %f %f\n", m.m1, m.m5, m.m9, m.m13);
        printf("%f %f %f %f\n", m.m2, m.m6, m.m10, m.m14);
        printf("%f %f %f %f\n", m.m3, m.m7, m.m11, m.m15);
    }
}

void PrintDebugInfo(AnimationState* animationState, Skeleton* skeleton, Node* nodes, int nodeCount, Matrix* boneTransforms) {
    PrintBoneMatrices(boneTransforms, skeleton);
    
    printf("\n--- Animation State ---\n");
    printf("Current Animation: %d\n", animationState->currentAnimation);
    printf("Is Playing: %s\n", animationState->isPlaying ? "Yes" : "No");
    printf("Current Time: %f\n", animationState->time);
    
    if (animationState->currentAnimation < animationState->animationCount) {
        AnimationData* currentAnim = &animationState->animations[animationState->currentAnimation];
        printf("Animation Name: %s\n", currentAnim->name);
        printf("Animation Duration: %f\n", currentAnim->duration);
    }
}

void PrintMeshData(CustomMesh* mesh) {
    printf("Vertex count: %d\n", mesh->vertexCount);
    printf("Index count: %d\n", mesh->indexCount);
    if (mesh->vertices) {
        printf("First 3 vertices:\n");
        for (int i = 0; i < 3 && i < mesh->vertexCount; i++) {
            printf("  (%f, %f, %f)\n", mesh->vertices[i*3], mesh->vertices[i*3+1], mesh->vertices[i*3+2]);
        }
    } else {
        printf("No vertex data\n");
    }
    if (mesh->texcoords) {
        printf("Has texture coordinates\n");
    } else {
        printf("No texture coordinates\n");
    }
    if (mesh->normals) {
        printf("Has normals\n");
    } else {
        printf("No normals\n");
    }
    if (mesh->boneIds) {
        printf("Has bone IDs\n");
    } else {
        printf("No bone IDs\n");
    }
    if (mesh->boneWeights) {
        printf("Has bone weights\n");
    } else {
        printf("No bone weights\n");
    }
}

void SetShaderValueMatrixArray(Shader shader, int locIndex, const Matrix* matrices, int count) {
    float* matrixData = (float*)malloc(count * 16 * sizeof(float));
    for (int i = 0; i < count; i++) {
        memcpy(matrixData + i * 16, &matrices[i], 16 * sizeof(float));
    }
    SetShaderValueV(shader, locIndex, matrixData, SHADER_UNIFORM_FLOAT, count * 16);
    free(matrixData);
}



Vector3 MatrixToTranslation(Matrix m) {
    return (Vector3){ m.m12, m.m13, m.m14 };
}

Vector3 MatrixToScale(Matrix m) {
    return (Vector3){
        Vector3Length((Vector3){ m.m0, m.m1, m.m2 }),
        Vector3Length((Vector3){ m.m4, m.m5, m.m6 }),
        Vector3Length((Vector3){ m.m8, m.m9, m.m10 })
    };
}

/*
Quaternion QuaternionFromMatrix(Matrix m) {
    Quaternion q;
    float trace = m.m0 + m.m5 + m.m10;
    if (trace > 0) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        q.w = 0.25f / s;
        q.x = (m.m6 - m.m9) * s;
        q.y = (m.m8 - m.m2) * s;
        q.z = (m.m1 - m.m4) * s;
    } else {
        if (m.m0 > m.m5 && m.m0 > m.m10) {
            float s = 2.0f * sqrtf(1.0f + m.m0 - m.m5 - m.m10);
            q.w = (m.m6 - m.m9) / s;
            q.x = 0.25f * s;
            q.y = (m.m4 + m.m1) / s;
            q.z = (m.m8 + m.m2) / s;
        } else if (m.m5 > m.m10) {
            float s = 2.0f * sqrtf(1.0f + m.m5 - m.m0 - m.m10);
            q.w = (m.m8 - m.m2) / s;
            q.x = (m.m4 + m.m1) / s;
            q.y = 0.25f * s;
            q.z = (m.m9 + m.m6) / s;
        } else {
            float s = 2.0f * sqrtf(1.0f + m.m10 - m.m0 - m.m5);
            q.w = (m.m1 - m.m4) / s;
            q.x = (m.m8 + m.m2) / s;
            q.y = (m.m9 + m.m6) / s;
            q.z = 0.25f * s;
        }
    }
    return q;
}
*/

void DrawSkeleton(CustomModel model, Matrix* boneTransforms) {
    for (int i = 0; i < model.boneCount; i++) {
        if (model.bones[i].parent >= 0) {
            Vector3 start = Vector3Transform((Vector3){0,0,0}, boneTransforms[i]);
            Vector3 end = Vector3Transform((Vector3){0,0,0}, boneTransforms[model.bones[i].parent]);
            DrawLine3D(start, end, RED);
        }
    }
}

Matrix TransformToMatrix(CustomTransform t) {
    Matrix translation = MatrixTranslate(t.translation.x, t.translation.y, t.translation.z);
    Matrix rotation = QuaternionToMatrix(t.rotation);
    Matrix scale = MatrixScale(t.scale.x, t.scale.y, t.scale.z);
    return MatrixMultiply(MatrixMultiply(scale, rotation), translation);
}

void UnloadCustomModel(CustomModel model) {
    for (int i = 0; i < model.meshCount; i++) {
        free(model.meshes[i].vertices);
        free(model.meshes[i].normals);
        free(model.meshes[i].texcoords);
        free(model.meshes[i].colors);
        free(model.meshes[i].tangents);
        free(model.meshes[i].indices);
        free(model.meshes[i].boneIds);
        free(model.meshes[i].boneWeights);

        // Delete VAO and VBOs
        rlUnloadVertexArray(model.meshesGPU[i].vaoId);
        for (int j = 0; j < 8; j++) {
            rlUnloadVertexBuffer(model.meshesGPU[i].vboId[j]);
        }
    }

    free(model.meshes);
    free(model.meshesGPU);
    free(model.meshMaterial);

    for (int i = 0; i < model.materialCount; i++) {
        UnloadTexture(model.materials[i].albedoMap);
        UnloadTexture(model.materials[i].normalMap);
        UnloadTexture(model.materials[i].metallicRoughnessMap);
        UnloadTexture(model.materials[i].emissiveMap);
        UnloadTexture(model.materials[i].occlusionMap);
        UnloadShader(model.materials[i].shader);
    }

    free(model.materials);
    free(model.bones);
    free(model.bindPose);
}


Material ConvertToRaylibMaterial(CustomMaterial* customMaterial) {
    Material material = LoadMaterialDefault();
    
    if (customMaterial == NULL) {
        printf("Error: customMaterial is NULL in ConvertToRaylibMaterial\n");
        return material;
    }

    material.shader = customMaterial->shader;
    material.maps[MATERIAL_MAP_DIFFUSE].color = customMaterial->color;
    
    if (customMaterial->albedoMap.id > 0) {
        material.maps[MATERIAL_MAP_DIFFUSE].texture = customMaterial->albedoMap;
    } else {
        printf("Warning: Invalid albedoMap texture in ConvertToRaylibMaterial\n");
    }

    if (customMaterial->normalMap.id > 0) {
        material.maps[MATERIAL_MAP_NORMAL].texture = customMaterial->normalMap;
    }

    if (customMaterial->metallicRoughnessMap.id > 0) {
        material.maps[MATERIAL_MAP_METALNESS].texture = customMaterial->metallicRoughnessMap;
        material.maps[MATERIAL_MAP_ROUGHNESS].texture = customMaterial->metallicRoughnessMap;
    }

    if (customMaterial->occlusionMap.id > 0) {
        material.maps[MATERIAL_MAP_OCCLUSION].texture = customMaterial->occlusionMap;
    }

    if (customMaterial->emissiveMap.id > 0) {
        material.maps[MATERIAL_MAP_EMISSION].texture = customMaterial->emissiveMap;
    }

    material.params[0] = customMaterial->metallic;
    material.params[1] = customMaterial->roughness;
    material.params[2] = customMaterial->emissiveFactor[0];
    material.params[3] = customMaterial->emissiveFactor[1];

    return material;
}


void DrawCustomModel(CustomModel model, Matrix transform, Matrix* boneTransforms, Camera camera) {
    printf("Entering DrawCustomModel\n");

    if (model.materialCount == 0 || model.materials == NULL) {
        printf("Error: No materials in the model\n");
        return;
    }

    // Create a basic shader
    Shader basicShader = LoadShaderFromMemory(basicVertexShader, basicFragmentShader);
    printf("Created basic shader with ID: %u\n", basicShader.id);

    // Get shader locations
    int mvpLoc = GetShaderLocation(basicShader, "mvp");
    int colorLoc = GetShaderLocation(basicShader, "colDiffuse");

    printf("Shader locations: mvp = %d, colDiffuse = %d\n", mvpLoc, colorLoc);

    printf("Calculating view projection matrix\n");
    Matrix viewProjection = GetCameraMatrix(camera);
    Matrix projection = MatrixPerspective(camera.fovy * DEG2RAD,
                                          (double)GetScreenWidth() / GetScreenHeight(),
                                          RL_CULL_DISTANCE_NEAR,
                                          RL_CULL_DISTANCE_FAR);
    viewProjection = MatrixMultiply(viewProjection, projection);

    printf("Looping through meshes (count: %d)\n", model.meshCount);
    for (int i = 0; i < model.meshCount; i++) {
        printf("Processing mesh %d\n", i);
        CustomMeshGPU* meshGPU = &model.meshesGPU[i];
        
        if (model.meshMaterial == NULL) {
            printf("Error: meshMaterial is NULL\n");
            continue;
        }
        
        int materialIndex = model.meshMaterial[i];
        if (materialIndex < 0 || materialIndex >= model.materialCount) {
            printf("Error: Invalid material index %d for mesh %d\n", materialIndex, i);
            continue;
        }

        printf("Setting shader uniforms\n");
        Matrix mvp = MatrixMultiply(transform, viewProjection);
        SetShaderValueMatrix(basicShader, mvpLoc, mvp);

        // Set a default color
        Vector4 color = { 1.0f, 1.0f, 1.0f, 1.0f };
        SetShaderValue(basicShader, colorLoc, &color, SHADER_UNIFORM_VEC4);

        printf("Checking mesh data:\n");
        printf("  Vertex count: %d\n", meshGPU->mesh.vertexCount);
        printf("  Triangle count: %d\n", meshGPU->mesh.triangleCount);
        printf("  Vertex buffer object ID: %u\n", meshGPU->mesh.vboId[0]);
        printf("  Vertex array object ID: %u\n", meshGPU->mesh.vaoId);

        printf("Drawing mesh\n");
        if (meshGPU->vaoId > 0) {
            BeginShaderMode(basicShader);
            DrawMesh(meshGPU->mesh, LoadMaterialDefault(), transform);
            EndShaderMode();
        } else {
            printf("Warning: Invalid VAO ID for mesh %d\n", i);
        }
    }

    printf("Unloading basic shader\n");
    UnloadShader(basicShader);

    printf("Exiting DrawCustomModel\n");
}

void UploadCustomMesh(CustomMesh* mesh, CustomMeshGPU* meshGPU) {
    printf("Entering UploadCustomMesh\n");

    if (mesh == NULL || meshGPU == NULL) {
        printf("Error: Null pointer in UploadCustomMesh\n");
        return;
    }

    printf("Mesh data: vertexCount = %d, indexCount = %d\n", mesh->vertexCount, mesh->indexCount);

    meshGPU->vertexCount = mesh->vertexCount;
    meshGPU->indexCount = mesh->indexCount;

    printf("Creating mesh for GPU\n");
    meshGPU->mesh.vertexCount = mesh->vertexCount;
    meshGPU->mesh.triangleCount = mesh->indexCount / 3;

    // Allocate memory and copy data only if it exists in the source mesh
    if (mesh->vertices) {
        printf("Allocating and copying vertices\n");
        meshGPU->mesh.vertices = (float *)RL_MALLOC(mesh->vertexCount * 3 * sizeof(float));
        if (meshGPU->mesh.vertices == NULL) {
            printf("Error: Failed to allocate memory for vertices\n");
            return;
        }
        memcpy(meshGPU->mesh.vertices, mesh->vertices, mesh->vertexCount * 3 * sizeof(float));
    } else {
        printf("Warning: No vertices in source mesh\n");
        meshGPU->mesh.vertices = NULL;
    }

    if (mesh->texcoords) {
        printf("Allocating and copying texcoords\n");
        meshGPU->mesh.texcoords = (float *)RL_MALLOC(mesh->vertexCount * 2 * sizeof(float));
        if (meshGPU->mesh.texcoords == NULL) {
            printf("Error: Failed to allocate memory for texcoords\n");
            return;
        }
        memcpy(meshGPU->mesh.texcoords, mesh->texcoords, mesh->vertexCount * 2 * sizeof(float));
    } else {
        printf("Warning: No texcoords in source mesh\n");
        meshGPU->mesh.texcoords = NULL;
    }

    if (mesh->normals) {
        printf("Allocating and copying normals\n");
        meshGPU->mesh.normals = (float *)RL_MALLOC(mesh->vertexCount * 3 * sizeof(float));
        if (meshGPU->mesh.normals == NULL) {
            printf("Error: Failed to allocate memory for normals\n");
            return;
        }
        memcpy(meshGPU->mesh.normals, mesh->normals, mesh->vertexCount * 3 * sizeof(float));
    } else {
        printf("Warning: No normals in source mesh\n");
        meshGPU->mesh.normals = NULL;
    }

    if (mesh->indices) {
        printf("Allocating and copying indices\n");
        meshGPU->mesh.indices = (unsigned short *)RL_MALLOC(mesh->indexCount * sizeof(unsigned short));
        if (meshGPU->mesh.indices == NULL) {
            printf("Error: Failed to allocate memory for indices\n");
            return;
        }
        memcpy(meshGPU->mesh.indices, mesh->indices, mesh->indexCount * sizeof(unsigned short));
    } else {
        printf("Warning: No indices in source mesh\n");
        meshGPU->mesh.indices = NULL;
    }

    meshGPU->mesh.colors = NULL;  // We're not using colors in this example
    meshGPU->mesh.animVertices = NULL;
    meshGPU->mesh.animNormals = NULL;
    meshGPU->mesh.boneIds = NULL;
    meshGPU->mesh.boneWeights = NULL;

    printf("Uploading mesh to GPU\n");
    UploadMesh(&meshGPU->mesh, false);

    printf("Mesh uploaded. VAO ID: %u\n", meshGPU->mesh.vaoId);

    printf("Exiting UploadCustomMesh\n");
}

// Main function
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_gltf_file>\n", argv[0]);
        return -1;
    }

    // Initialize Raylib
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1280, 720, "Custom glTF Loader with Animations");
    SetTargetFPS(60);

    printf("Raylib initialized\n");

    // Load glTF data
    cgltf_data* data = LoadGLTFFile(argv[1]);
    if (!data) {
        printf("Failed to load glTF file.\n");
        CloseWindow();
        return -1;
    }

    printf("glTF file loaded successfully\n");

    // Load nodes
    Node nodes[MAX_NODES] = {0};
    int nodeCount = 0;
    LoadNodes(data, nodes, &nodeCount);
    BuildNodeHierarchy(nodes, nodeCount, data);

    printf("Nodes loaded and hierarchy built\n");

    // Print node hierarchy for debugging
    printf("\n--- Node Hierarchy ---\n");
    for (int i = 0; i < nodeCount; ++i) {
        if (nodes[i].parentIndex == -1) {
            PrintNodeHierarchy(nodes, i, 0);
        }
    }

    // Load skeleton
    Skeleton skeleton = {0};
    LoadSkeleton(data, &skeleton, nodes, nodeCount);

    printf("Skeleton loaded\n");

    // Load animations
    AnimationState animationState = {0};
    LoadAnimations(data, &animationState, nodes, nodeCount);
    animationState.isPlaying = true;

    printf("Animations loaded\n");

    // Load model
    TextureCache textureCache = {0};
    MyBoneInfo* myBoneInfos = NULL;
    Matrix* inverseBindPose = NULL;
    CustomModel model = LoadCustomModelFromGLTF(data, &skeleton, &textureCache, nodes, nodeCount, &myBoneInfos, &inverseBindPose);

    printf("CustomModel loaded\n");

    // Create boneTransforms array
    Matrix* boneTransforms = (Matrix*)malloc(sizeof(Matrix) * skeleton.boneCount);
    if (!boneTransforms) {
        printf("Failed to allocate memory for bone transforms.\n");
        CloseWindow();
        return -1;
    }
    for (int i = 0; i < skeleton.boneCount; ++i) {
        boneTransforms[i] = MatrixIdentity();
    }

    printf("Bone transforms initialized\n");

    // Camera setup
    Camera camera = { 0 };
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    printf("Camera set up\n");

    // Model transformation
    Vector3 modelPosition = { 0.0f, 0.0f, 0.0f };
    float modelScale = 1.0f;

    printf("Starting main game loop\n");

    // Main game loop
    while (!WindowShouldClose()) {
        printf("Frame start\n");

        // Update
        printf("Updating camera\n");
        UpdateCamera(&camera, CAMERA_ORBITAL);

        // Handle input
        printf("Handling input\n");
        if (IsKeyDown(KEY_RIGHT)) modelPosition.x += 0.1f;
        if (IsKeyDown(KEY_LEFT)) modelPosition.x -= 0.1f;
        if (IsKeyDown(KEY_UP)) modelPosition.y += 0.1f;
        if (IsKeyDown(KEY_DOWN)) modelPosition.y -= 0.1f;
        if (IsKeyDown(KEY_W)) modelPosition.z += 0.1f;
        if (IsKeyDown(KEY_S)) modelPosition.z -= 0.1f;
        if (IsKeyDown(KEY_A)) modelScale *= 1.01f;
        if (IsKeyDown(KEY_D)) modelScale *= 0.99f;

        if (IsKeyPressed(KEY_SPACE)) {
            animationState.isPlaying = !animationState.isPlaying;
        }

        if (IsKeyPressed(KEY_RIGHT)) {
            animationState.currentAnimation = (animationState.currentAnimation + 1) % animationState.animationCount;
            animationState.time = 0.0f;
        }

        // Update animation
        printf("Updating animation\n");
        if (animationState.isPlaying && animationState.animationCount > 0) {
            animationState.time += GetFrameTime();
            AnimationData* currentAnimation = &animationState.animations[animationState.currentAnimation];
            if (animationState.time > currentAnimation->duration) {
                animationState.time = fmodf(animationState.time, currentAnimation->duration);
            }
            printf("Calling UpdateAnimation\n");
            UpdateAnimation(&animationState, nodes, nodeCount);
            printf("Calling ApplyNodeTransforms\n");
            ApplyNodeTransforms(nodes, nodeCount, &skeleton, myBoneInfos, inverseBindPose, boneTransforms);
        }

        // Draw
        printf("Beginning drawing\n");
        BeginDrawing();
        ClearBackground(RAYWHITE);

        printf("Beginning 3D mode\n");
        BeginMode3D(camera);

        // Draw grid
        printf("Drawing grid\n");
        DrawGrid(10, 1.0f);

        // Draw model
        printf("Calculating model transform\n");
        Matrix transform = MatrixMultiply(
            MatrixTranslate(modelPosition.x, modelPosition.y, modelPosition.z),
            MatrixScale(modelScale, modelScale, modelScale)
        );
        printf("Drawing custom model\n");
        DrawCustomModel(model, transform, boneTransforms, camera);

        // Draw skeleton (for debugging)
        printf("Drawing skeleton\n");
        DrawSkeleton(model, boneTransforms);

        printf("Ending 3D mode\n");
        EndMode3D();

        // Draw UI
        printf("Drawing UI\n");
        DrawFPS(10, 10);
        DrawText(TextFormat("Animation: %s", animationState.animations[animationState.currentAnimation].name), 10, 30, 20, BLACK);
        DrawText(TextFormat("Time: %.2f / %.2f", animationState.time, animationState.animations[animationState.currentAnimation].duration), 10, 50, 20, BLACK);
        DrawText("Use arrow keys to move model", 10, 70, 20, BLACK);
        DrawText("Use A/D to scale model", 10, 90, 20, BLACK);
        DrawText("Press SPACE to play/pause animation", 10, 110, 20, BLACK);
        DrawText("Press RIGHT to change animation", 10, 130, 20, BLACK);

        printf("Ending drawing\n");
        EndDrawing();

        printf("Frame end\n");
    }

    printf("Exiting main game loop\n");
    // De-Initialization
    UnloadCustomModel(model);
    free(myBoneInfos);
    free(inverseBindPose);
    free(boneTransforms);
    FreeAnimationData(&animationState, nodeCount);
    FreeGLTFData(data);

    printf("Resources freed\n");

    CloseWindow();

    printf("Window closed\n");

    return 0;
}
