#ifndef DOLPHIN_GXEXTRA_H
#define DOLPHIN_GXEXTRA_H
// Extra types for PC
#ifdef TARGET_PC
#include <dolphin/gx/GXStruct.h>
#include <dolphin/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float r;
  float g;
  float b;
  float a;
} GXColorF32;

void GXDestroyTexObj(GXTexObj* obj);
// Marks a texture object as streamed (or clears the mark): a program that rewrites the object's image in place
// regularly, such as a movie's frames, and announces each new image with GXInitTexObjData. A streamed object keeps a
// texture of its own, rewritten when a new image's bytes differ, instead of a new texture for each image, and its
// images stay out of the cache of textures shared by content. GXInitTexObj clears the mark.
void GXInitTexObjStreaming(GXTexObj* obj, GXBool streaming);
// Prepares the texture object's image for the host's GPU on background threads, from a copy of its data taken now:
// they convert it and create its texture, so that the first draw with it only uploads it. For an image that does not
// change afterwards (one loaded from disc); palette formats are left to their first draw.
void GXPrepareTexObj(const GXTexObj* obj);
void GXDestroyTlutObj(GXTlutObj* obj);
void GXDestroyCopyTex(void* dest);

void GXColor4f32(float r, float g, float b, float a);

#ifdef __cplusplus
}
#endif
#endif

#endif
