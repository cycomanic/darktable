/*
    This file is part of darktable,
    Copyright (C) 2010-2024 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common/darktable.h"
#include "common/gaussian.h"
#include "common/math.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "dtgtk/drawingarea.h"
#include "dtgtk/gradientslider.h"
#include "dtgtk/togglebutton.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "common/curve_tools.h"


#include <librsvg/rsvg.h>
// ugh, ugly hack. why do people break stuff all the time?
#ifndef RSVG_CAIRO_H
#include <librsvg/rsvg-cairo.h>
#endif


DT_MODULE_INTROSPECTION(1, dt_iop_zonesystem_params_t)
#define MAX_ZONE_SYSTEM_SIZE 24
#define DEGREE 3

/** gui params. */
typedef struct dt_iop_zonesystem_params_t
{
  int size; // $DEFAULT: 10
  float zone[MAX_ZONE_SYSTEM_SIZE + 1]; // $DEFAULT: -1.0
} dt_iop_zonesystem_params_t;

/** and pixelpipe data is just the same */
typedef struct dt_iop_zonesystem_data_t
{
  dt_iop_zonesystem_params_t params;
  float rzscale;
  float zonemap_offset[MAX_ZONE_SYSTEM_SIZE];
  float zonemap_scale[MAX_ZONE_SYSTEM_SIZE];
  float *curve;           // High-resolution curve for processing
  int curve_size;         // Size of the curve array
} dt_iop_zonesystem_data_t;

typedef struct dt_iop_zonesystem_global_data_t
{
  int kernel_zonesystem;
} dt_iop_zonesystem_global_data_t;


typedef struct dt_iop_zonesystem_gui_data_t
{
  guchar *in_preview_buffer;
  guchar *out_preview_buffer;
  int preview_width, preview_height;
  GtkWidget *preview;
  GtkWidget *zones;
  float press_x, press_y, mouse_x, mouse_y;
  gboolean hilite_zone;
  gboolean is_dragging;
  int current_zone;
  int zone_under_mouse;
  int mouse_over_output_zones;

  cairo_surface_t *image;
  guint8 *image_buffer;
  int image_width, image_height;

} dt_iop_zonesystem_gui_data_t;

typedef struct {
    float x;
    float y;
} Point;

const char *name()
{
  return _("zone system");
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_ALLOW_TILING
         | IOP_FLAGS_PREVIEW_NON_OPENCL;
}

const char *deprecated_msg()
{
  return NULL;  // Return NULL to hide the deprecated message
}

int default_group()
{
  return IOP_GROUP_TONE | IOP_GROUP_GRADING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

/* get the zone index of pixel lightness from zonemap */
static inline int _iop_zonesystem_zone_index_from_lightness(float lightness, float *zonemap, int size)
{
  for(int k = 0; k < size - 1; k++)
    if(zonemap[k + 1] >= lightness) return k;
  return size - 1;
}

// Helper function to calculate basis functions
static void _basis_functions(int degree, float t, int num_control_points,
                            const float *knots, int knots_count, float *N) {
    float *temp = (float *)malloc((knots_count - 1) * sizeof(float));

    // Initialize first order basis functions N[i][1]
    for (int i = 0; i < knots_count - 1; i++) {
        if (t >= knots[i] && t < knots[i + 1]) {
            temp[i] = 1.0f;
        } else {
            temp[i] = 0.0f;
        }
    }

    // Calculate higher order basis functions recursively
    for (int p = 2; p <= degree + 1; p++) {
        for (int i = 0; i < knots_count - p; i++) {
            float d = 0.0f;
            if (temp[i] != 0.0f) {
                float denom = knots[i + p - 1] - knots[i];
                if (denom != 0.0f) {
                    d = ((t - knots[i]) * temp[i]) / denom;
                }
            }

            float e = 0.0f;
            if (temp[i + 1] != 0.0f) {
                float denom = knots[i + p] - knots[i + 1];
                if (denom != 0.0f) {
                    e = ((knots[i + p] - t) * temp[i + 1]) / denom;
                }
            }

            temp[i] = d + e;
        }
    }

    // Handle the last point
    if (fabs(t - knots[knots_count - 1]) < 1e-10f) {
        temp[num_control_points - 1] = 1.0f;
    }

    // Copy to output
    for (int i = 0; i < num_control_points; i++) {
        N[i] = temp[i];
    }

    free(temp);
}

float _nurbs_evaluate(const Point *control_points, const float *weights,
                      int num_control_points, const float *knots, int knots_count,
                      int degree, float target_x) {
    // Clamp target_x to valid range
    if (target_x < control_points[0].x) {
        target_x = control_points[0].x;
    }
    if (target_x > control_points[num_control_points - 1].x) {
        target_x = control_points[num_control_points - 1].x;
    }

    float max_t = knots[knots_count - 1];
    float t_min = 0.0f;
    float t_max = max_t;
    float tolerance = 1e-6f;
    int max_iterations = 20;
    float t = max_t / 2.0f; // Initial guess

    for (int iter = 0; iter < max_iterations; iter++) {
        t = (t_min + t_max) / 2.0f;

        // Calculate basis functions
        float *N = (float *)malloc(num_control_points * sizeof(float));
        _basis_functions(degree, t, num_control_points, knots, knots_count, N);

        // Calculate weighted sum
        float weight_sum = 0.0f;
        for (int i = 0; i < num_control_points; i++) {
            weight_sum += N[i] * weights[i];
        }

        // Calculate x(t) and y(t)
        float x_at_t = 0.0f;
        float y_at_t = 0.0f;

        if (weight_sum != 0.0f) {
            for (int i = 0; i < num_control_points; i++) {
                float rational_basis = (N[i] * weights[i]) / weight_sum;
                x_at_t += rational_basis * control_points[i].x;
                y_at_t += rational_basis * control_points[i].y;
            }
        }

        // Check if we found target_x
        if (fabs(x_at_t - target_x) < tolerance) {
            free(N);
            return y_at_t;
        }

        // Adjust search range (binary search)
        if (x_at_t < target_x) {
            t_min = t;
        } else {
            t_max = t;
        }

        free(N);
    }

    // After max iterations, use the final t value
    // This handles the edge case where we didn't converge exactly
    float *N = (float *)malloc(num_control_points * sizeof(float));
    _basis_functions(degree, t, num_control_points, knots, knots_count, N);

    float weight_sum = 0.0f;
    float y = 0.0f;

    for (int i = 0; i < num_control_points; i++) {
        weight_sum += N[i] * weights[i];
    }

    if (weight_sum != 0.0f) {
        for (int i = 0; i < num_control_points; i++) {
            float rational_basis = (N[i] * weights[i]) / weight_sum;
            y += rational_basis * control_points[i].y;
        }
    }

    free(N);
    return y;
}

/* Create clamped knot vector for cubic B-spline (degree 3) */
static inline void _create_clamped_knots(int num_control_points, int degree, float *knots)
{
  int knots_count = num_control_points + degree + 1;
  knots[0] = 0.0f;
  for (int i = 1; i < knots_count ; i++) {
      if ((i > degree ) && (i < num_control_points + 1))
      {
          knots[i] = knots[i - 1] + 1;
      }
      else
       {
                knots[i] = knots[i - 1];
       }
   }
}

/* calculate a zonemap with scale values for each zone based on NURBS curve from control points */
/* curve_array should be at least 10x larger than zonemap size and minimum 200 entries */
static inline void _iop_zonesystem_calculate_zonemap(dt_iop_zonesystem_params_t *p, float *zonemap, float *curve_array, int curve_size)
{
  /* Count actual control points (excluding -1 placeholders) */
  int num_control_points = 0; /* the outer points are always control points */
  Point *control_points = malloc(p->size * sizeof(Point));

  control_points[0] = (Point){0.0f, 0.0f};
  num_control_points++;
  for (int k = 1; k < p->size-1; k++)
  {
    if (p->zone[k] != -1)
    {
      control_points[num_control_points] = (Point){(float)k/(p->size - 1), p->zone[k]};
      num_control_points++;
    }
  }
  control_points[num_control_points] = (Point){1.0f, 1.0f};
  num_control_points++;

  /* If fewer than 4 control points, fall back to linear interpolation */
  if (num_control_points < 4)
  {
    int steps = 0;
    int pk = 0;

    /* Fill zonemap with linear interpolation */
    for (int k = 0; k < p->size; k++)
    {
      if ((k > 0 && k < p->size - 1) && p->zone[k] == -1)
        steps++;
      else
      {
        zonemap[k] = k == 0 ? 0.0f : k == (p->size - 1) ? 1.0f : p->zone[k];
        for (int l = 1; l <= steps; l++)
          zonemap[pk + l] = zonemap[pk] + (((zonemap[k] - zonemap[pk]) / (steps + 1)) * l);
        pk = k;
        steps = 0;
      }
    }

    /* Fill curve_array with linear interpolation */
    if (curve_array && curve_size > 0)
    {
      for (int i = 0; i < curve_size; i++)
      {
        float u = (float)i / (float)(curve_size - 1);
        /* Find which control points bracket this u value */
        int idx = 0;
        for (int j = 0; j < num_control_points - 1; j++)
        {
          if (u >= control_points[j].x && u <= control_points[j + 1].x)
          {
            idx = j;
            break;
          }
        }

        /* Linear interpolation between control points */
        float t = (u - control_points[idx].x) / (control_points[idx + 1].x - control_points[idx].x);
        curve_array[i] = control_points[idx].y + t * (control_points[idx + 1].y - control_points[idx].y);
        curve_array[i] = fmaxf(0.0f, fminf(1.0f, curve_array[i]));
      }
    }

    free(control_points);
    return;
  }

  /* Use NURBS curve with cubic spline (degree 3) */
  int degree = 2;
  int order = degree + 1;
  int knots_count = num_control_points + order;
  float *knots = malloc(knots_count * sizeof(float));
  float *weights = malloc(num_control_points * sizeof(float));

  /* Set up control points and weights */
  for (int i = 0; i < num_control_points; i++)
  {
    /* Edge points have weight 1, internal points have weight 10 */
    weights[i] = (i == 0 || i == num_control_points - 1) ? 1.0f : 10.0f;
  }

  /* Create clamped knot vector */
  _create_clamped_knots(num_control_points, degree, knots);

  /* Evaluate NURBS curve and fill zonemap (for GUI display) */
  for (int k = 0; k < p->size; k++)
  {
    /* Map zone index k to parameter u in [0, 1] */
    float u = (float)k/((float) p->size -1);
    zonemap[k] = _nurbs_evaluate(control_points, weights, num_control_points, knots, knots_count, degree, u);
    /* Clamp to valid range */
    zonemap[k] = fmaxf(0.0f, fminf(1.0f, zonemap[k]));
  }

  /* Evaluate NURBS curve and fill high-resolution curve_array (for processing) */
  if (curve_array && curve_size > 0)
  {
    for (int i = 0; i < curve_size; i++)
    {
      /* Map curve index i to parameter u in [0, 1] */
      float u = (float)i / (float)(curve_size - 1);
      curve_array[i] = _nurbs_evaluate(control_points, weights, num_control_points, knots, knots_count, degree, u);
      /* Clamp to valid range */
      curve_array[i] = fmaxf(0.0f, fminf(1.0f, curve_array[i]));
    }
  }

  free(knots);
  free(control_points);
  free(weights);
}

static void process_common_setup(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                                 const void *const ivoid, void *const ovoid, const dt_iop_roi_t *const roi_in,
                                 const dt_iop_roi_t *const roi_out)
{
  const int width = roi_out->width;
  const int height = roi_out->height;

  if(self->dev->gui_attached && (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW))
  {
    dt_iop_zonesystem_gui_data_t *g = self->gui_data;
    dt_iop_gui_enter_critical_section(self);
    if(g->in_preview_buffer == NULL || g->out_preview_buffer == NULL || g->preview_width != width
       || g->preview_height != height)
    {
      g_free(g->in_preview_buffer);
      g_free(g->out_preview_buffer);
      g->in_preview_buffer = g_malloc_n((size_t)width * height, sizeof(guchar));
      g->out_preview_buffer = g_malloc_n((size_t)width * height, sizeof(guchar));
      g->preview_width = width;
      g->preview_height = height;
    }
    dt_iop_gui_leave_critical_section(self);
  }
}

static void process_common_cleanup(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                                   const void *const ivoid, void *const ovoid,
                                   const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_zonesystem_data_t *d = piece->data;
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;

  const int width = roi_out->width;
  const int height = roi_out->height;
  const size_t ch = piece->colors;
  const int size = d->params.size;

  if(piece->pipe->mask_display & DT_DEV_PIXELPIPE_DISPLAY_MASK) dt_iop_alpha_copy(ivoid, ovoid, width, height);

  /* if gui and have buffer lets gaussblur and fill buffer with zone indexes */
  if(self->dev->gui_attached
     && (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
     && g && g->in_preview_buffer
     && g->out_preview_buffer)
  {

    float Lmax[] = { 100.0f };  // Set to 100 for 0-100 range
    float Lmin[] = { 0.0f };

    /* setup gaussian kernel */
    const int radius = 8;
    const float sigma = 2.5 * (radius * roi_in->scale / piece->iscale);

    dt_gaussian_t *gauss = dt_gaussian_init(width, height, 1, Lmax, Lmin, sigma, DT_IOP_GAUSSIAN_ZERO);

    float *tmp = g_malloc_n((size_t)width * height, sizeof(float));

    if(gauss && tmp)
    {

      // Calculate luminance from linear RGB for input
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)width * height; k++)
      {
        const float *pixel = &((float *)ivoid)[ch * k];
        tmp[k] = 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
        tmp[k] *= 100.0f;  // Scale to 0-100 range
      }

      dt_gaussian_blur(gauss, tmp, tmp);

      /* create zonemap preview for input */
      dt_iop_gui_enter_critical_section(self);
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)width * height; k++)
      {
        g->in_preview_buffer[k] = CLAMPS(tmp[k] * d->rzscale, 0, size - 2);
      }
      dt_iop_gui_leave_critical_section(self);


      // Calculate luminance from linear RGB for output
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)width * height; k++)
      {
        const float *pixel = &((float *)ovoid)[ch * k];
        tmp[k] = 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
        tmp[k] *= 100.0f;  // Scale to 0-100 range
      }

      dt_gaussian_blur(gauss, tmp, tmp);

      /* create zonemap preview for output */
      dt_iop_gui_enter_critical_section(self);
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)width * height; k++)
      {
        g->out_preview_buffer[k] = CLAMPS(tmp[k] * d->rzscale, 0, size - 2);
      }
      dt_iop_gui_leave_critical_section(self);
      
    }

    g_free(tmp);
    if(gauss) dt_gaussian_free(gauss);
  }
  
}

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_zonesystem_data_t *const d = (const dt_iop_zonesystem_data_t *const)piece->data;
  process_common_setup(self, piece, ivoid, ovoid, roi_in, roi_out);

  const float *const restrict in = (const float *const)ivoid;
  float *const restrict out = (float *const)ovoid;
  const size_t npixels = (size_t)roi_out->width * roi_out->height;

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
  {
    const size_t k = i * 4;

    /* Calculate relative luminance from linear RGB */
    const float Y = 0.2126f * in[k] + 0.7152f * in[k+1] + 0.0722f * in[k+2];

    /* Look up target value from high-resolution curve */
    const float curve_position = Y * (d->curve_size - 1);
    const int curve_idx_low = CLAMPS((int)curve_position, 0, d->curve_size - 2);
    const int curve_idx_high = curve_idx_low + 1;

    /* Calculate interpolation factor (fractional part) */
    const float t = curve_position - curve_idx_low;

    /* Linear interpolation between two curve points */
    const float target_Y = d->curve[curve_idx_low] + t * (d->curve[curve_idx_high] - d->curve[curve_idx_low]);
    /* Calculate scaling factor */
    const float scale = (Y > 0.0001f) ? (target_Y / Y) : 1.0f;

    /* Apply scaling to RGB channels */
    out[k] = in[k] * scale;
    out[k+1] = in[k+1] * scale;
    out[k+2] = in[k+2] * scale;
    out[k+3] = in[k+3];
  }

  process_common_cleanup(self, piece, ivoid, ovoid, roi_in, roi_out);
}

//#ifdef HAVE_OPENCL
//int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
//               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
//{
//  dt_iop_zonesystem_data_t *data = piece->data;
//  dt_iop_zonesystem_global_data_t *gd = self->global_data;
//  cl_mem dev_zmo, dev_zms = NULL;
//  cl_int err = DT_OPENCL_DEFAULT_ERROR;
//
//  const int devid = piece->pipe->devid;
//  const int width = roi_in->width;
//  const int height = roi_in->height;
//
//  /* calculate zonemap */
//  const int size = data->params.size;
//  float zonemap[MAX_ZONE_SYSTEM_SIZE] = { -1 };
//  float zonemap_offset[ROUNDUP(MAX_ZONE_SYSTEM_SIZE, 16)] = { -1 };
//  float zonemap_scale[ROUNDUP(MAX_ZONE_SYSTEM_SIZE, 16)] = { -1 };
//
//  _iop_zonesystem_calculate_zonemap(&(data->params), zonemap, NULL, 0);
//
//  /* precompute scale and offset - adjusted for 0-1 range instead of 0-100 */
//  for(int k = 0; k < size - 1; k++) zonemap_scale[k] = (zonemap[k + 1] - zonemap[k]) * (size - 1);
//  for(int k = 0; k < size - 1; k++) zonemap_offset[k] = (k + 1) * zonemap[k] - k * zonemap[k + 1];
//
//  dev_zmo = dt_opencl_copy_host_to_device_constant(devid, sizeof(float) * ROUNDUP(MAX_ZONE_SYSTEM_SIZE, 16),
//                                                   zonemap_offset);
//  if(dev_zmo == NULL) goto error;
//  dev_zms = dt_opencl_copy_host_to_device_constant(devid, sizeof(float) * ROUNDUP(MAX_ZONE_SYSTEM_SIZE, 16),
//                                                   zonemap_scale);
//  if(dev_zms == NULL) goto error;
//
//  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_zonesystem, width, height,
//    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(size), CLARG(dev_zmo), CLARG(dev_zms));
//
//error:
//  dt_opencl_release_mem_object(dev_zmo);
//  dt_opencl_release_mem_object(dev_zms);
//  return err;
//}
//#endif


void init_global(dt_iop_module_so_t *self)
{
  const int program = 2; // basic.cl, from programs.conf
  dt_iop_zonesystem_global_data_t *gd = malloc(sizeof(dt_iop_zonesystem_global_data_t));
  self->data = gd;
  gd->kernel_zonesystem = dt_opencl_create_kernel(program, "zonesystem");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_zonesystem_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_zonesystem);
  free(self->data);
  self->data = NULL;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_zonesystem_params_t *p = (dt_iop_zonesystem_params_t *)p1;
  dt_iop_zonesystem_data_t *d = piece->data;

  d->params = *p;
  d->rzscale = (d->params.size - 1) / 100.0f;

  /* Allocate curve array (at least 10x zonemap size, minimum 200) */
  d->curve_size = MAX(200, d->params.size * 10);
  if(d->curve) free(d->curve);
  d->curve = malloc(d->curve_size * sizeof(float));

  /* Calculate zonemap and high-res curve */
  float zonemap[MAX_ZONE_SYSTEM_SIZE] = { -1 };
  _iop_zonesystem_calculate_zonemap(&(d->params), zonemap, d->curve, d->curve_size);
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_zonesystem_data_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
 dt_iop_zonesystem_data_t *d = piece->data;
  if(d && d->curve) free(d->curve);
  free(piece->data);
  piece->data = NULL;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  gtk_widget_queue_draw(GTK_WIDGET(g->zones));
}

static void _iop_zonesystem_redraw_preview_callback(gpointer instance,
                                                    dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_preview_draw(GtkWidget *widget, cairo_t *crf,
                                               dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_bar_draw(GtkWidget *widget, cairo_t *crf,
                                           dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_bar_motion_notify(GtkWidget *widget, GdkEventMotion *event,
                                                    dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_bar_leave_notify(GtkWidget *widget, GdkEventCrossing *event,
                                                   dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_bar_button_press(GtkWidget *widget, GdkEventButton *event,
                                                   dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_bar_button_release(GtkWidget *widget, GdkEventButton *event,
                                                     dt_iop_module_t *self);
static gboolean dt_iop_zonesystem_bar_scrolled(GtkWidget *widget, GdkEventScroll *event,
                                               dt_iop_module_t *self);


static void size_allocate_callback(GtkWidget *widget, GtkAllocation *allocation, dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;

  if(g->image) cairo_surface_destroy(g->image);
  free(g->image_buffer);

  /* load the dt logo as a background */
  g->image = dt_util_get_logo(MIN(allocation->width, allocation->height) * 0.75);
  if(g->image)
  {
    g->image_buffer = cairo_image_surface_get_data(g->image);
    g->image_width = cairo_image_surface_get_width(g->image);
    g->image_height = cairo_image_surface_get_height(g->image);
  }
  else
  {
    g->image_buffer = NULL;
    g->image_width = 0;
    g->image_height = 0;
  }
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = IOP_GUI_ALLOC(zonesystem);
  g->in_preview_buffer = g->out_preview_buffer = NULL;
  g->is_dragging = FALSE;
  g->hilite_zone = FALSE;
  g->preview_width = g->preview_height = 0;
  g->mouse_over_output_zones = FALSE;

  g->preview = dtgtk_drawing_area_new_with_height(0);
  g_signal_connect(G_OBJECT(g->preview), "size-allocate", G_CALLBACK(size_allocate_callback), self);
  g_signal_connect(G_OBJECT(g->preview), "draw", G_CALLBACK(dt_iop_zonesystem_preview_draw), self);
  gtk_widget_add_events(GTK_WIDGET(g->preview), GDK_POINTER_MOTION_MASK
                                                | GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK
                                                | GDK_LEAVE_NOTIFY_MASK);

  /* create the zonesystem bar widget */
  g->zones = gtk_drawing_area_new();
  gtk_widget_set_tooltip_text(g->zones, _("lightness zones\nuse mouse scrollwheel to change the number of zones\n"
                                          "left-click on a border to create a marker\n"
                                          "right-click on a marker to delete it"));
  g_signal_connect(G_OBJECT(g->zones), "draw", G_CALLBACK(dt_iop_zonesystem_bar_draw), self);
  g_signal_connect(G_OBJECT(g->zones), "motion-notify-event", G_CALLBACK(dt_iop_zonesystem_bar_motion_notify),
                   self);
  g_signal_connect(G_OBJECT(g->zones), "leave-notify-event", G_CALLBACK(dt_iop_zonesystem_bar_leave_notify),
                   self);
  g_signal_connect(G_OBJECT(g->zones), "button-press-event", G_CALLBACK(dt_iop_zonesystem_bar_button_press),
                   self);
  g_signal_connect(G_OBJECT(g->zones), "button-release-event",
                   G_CALLBACK(dt_iop_zonesystem_bar_button_release), self);
  g_signal_connect(G_OBJECT(g->zones), "scroll-event", G_CALLBACK(dt_iop_zonesystem_bar_scrolled), self);
  gtk_widget_add_events(GTK_WIDGET(g->zones), GDK_POINTER_MOTION_MASK
                                              | GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK
                                              | GDK_LEAVE_NOTIFY_MASK | darktable.gui->scroll_mask);
  gtk_widget_set_size_request(g->zones, -1, DT_PIXEL_APPLY_DPI(40));

  self->widget = dt_gui_vbox(g->preview, g->zones);

  /* add signal handler for preview pipe finish to redraw the preview */
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _iop_zonesystem_redraw_preview_callback);

  g->image = NULL;
  g->image_buffer = NULL;
  g->image_width = 0;
  g->image_height = 0;
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  g_free(g->in_preview_buffer);
  g_free(g->out_preview_buffer);
  if(g->image) cairo_surface_destroy(g->image);
  free(g->image_buffer);
}

#define DT_ZONESYSTEM_INSET DT_PIXEL_APPLY_DPI(5)
#define DT_ZONESYSTEM_BAR_SPLIT_WIDTH 0.0
#define DT_ZONESYSTEM_REFERENCE_SPLIT 0.30
static gboolean dt_iop_zonesystem_bar_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  dt_iop_zonesystem_params_t *p = self->params;

  const int inset = DT_ZONESYSTEM_INSET;
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width, height = allocation.height;
  cairo_surface_t *cst = dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);

  /* clear background */
  cairo_set_source_rgb(cr, .15, .15, .15);
  cairo_paint(cr);


  /* translate and scale */
  width -= 2 * inset;
  height -= 2 * inset;
  cairo_save(cr);
  cairo_translate(cr, inset, inset);
  cairo_scale(cr, width, height);

  /* render the bars */
  float zonemap[MAX_ZONE_SYSTEM_SIZE] = { 0 };
  _iop_zonesystem_calculate_zonemap(p, zonemap, NULL, 0);
  float s = (1. / (p->size - 2));
  cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
  for(int i = 0; i < p->size - 1; i++)
  {
    /* draw the reference zone */
    float z = s * i;
    cairo_rectangle(cr, (1. / (p->size - 1)) * i, 0, (1. / (p->size - 1)),
                    DT_ZONESYSTEM_REFERENCE_SPLIT - DT_ZONESYSTEM_BAR_SPLIT_WIDTH);
    cairo_set_source_rgb(cr, z, z, z);
    cairo_fill(cr);

    /* draw zone mappings */
    cairo_rectangle(cr, zonemap[i], DT_ZONESYSTEM_REFERENCE_SPLIT + DT_ZONESYSTEM_BAR_SPLIT_WIDTH,
                    (zonemap[i + 1] - zonemap[i]), 1.0 - DT_ZONESYSTEM_REFERENCE_SPLIT);
    cairo_set_source_rgb(cr, z, z, z);
    cairo_fill(cr);
  }
  cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
  cairo_restore(cr);

  /* render zonebar control lines */
  cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
  cairo_set_line_width(cr, 1.);
  cairo_rectangle(cr, inset, inset, width, height);
  cairo_set_source_rgb(cr, .1, .1, .1);
  cairo_stroke(cr);
  cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);

  /* render control points handles */
  cairo_set_source_rgb(cr, 0.6, 0.6, 0.6);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.));
  const float arrw = DT_PIXEL_APPLY_DPI(7.0f);
  for(int k = 1; k < p->size - 1; k++)
  {
    float nzw = zonemap[k + 1] - zonemap[k];
    float pzw = zonemap[k] - zonemap[k - 1];
    if((((g->mouse_x / width) > (zonemap[k] - (pzw / 2.0)))
        && ((g->mouse_x / width) < (zonemap[k] + (nzw / 2.0)))) || p->zone[k] != -1)
    {
      gboolean is_under_mouse = ((width * zonemap[k]) - arrw * .5f < g->mouse_x
                                 && (width * zonemap[k]) + arrw * .5f > g->mouse_x);

      cairo_move_to(cr, inset + (width * zonemap[k]), height + (2 * inset) - 1);
      cairo_rel_line_to(cr, -arrw * .5f, 0);
      cairo_rel_line_to(cr, arrw * .5f, -arrw);
      cairo_rel_line_to(cr, arrw * .5f, arrw);
      cairo_close_path(cr);

      if(is_under_mouse)
        cairo_fill(cr);
      else
        cairo_stroke(cr);
    }
  }


  /* push mem surface into widget */
  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);

  return TRUE;
}

static gboolean dt_iop_zonesystem_bar_button_press(GtkWidget *widget, GdkEventButton *event,
                                                   dt_iop_module_t *self)
{
  dt_iop_zonesystem_params_t *p = self->params;
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  const int inset = DT_ZONESYSTEM_INSET;
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width - 2 * inset; /*, height = allocation.height - 2*inset;*/

  /* calculate zonemap */
  float zonemap[MAX_ZONE_SYSTEM_SIZE] = { -1 };
  _iop_zonesystem_calculate_zonemap(p, zonemap,  NULL, 0);

  /* translate mouse into zone index */
  int k = _iop_zonesystem_zone_index_from_lightness(g->mouse_x / width, zonemap, p->size);
  float zw = zonemap[k + 1] - zonemap[k];
  if((g->mouse_x / width) > zonemap[k] + (zw / 2)) k++;


  if(event->button == GDK_BUTTON_PRIMARY)
  {
    if(p->zone[k] == -1)
    {
      p->zone[k] = zonemap[k];
      dt_dev_add_history_item(darktable.develop, self, TRUE);
    }
    g->is_dragging = TRUE;
    g->current_zone = k;
  }
  else if(event->button == GDK_BUTTON_SECONDARY)
  {
    /* clear the controlpoint */
    p->zone[k] = -1;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }

  return TRUE;
}

static gboolean dt_iop_zonesystem_bar_button_release(GtkWidget *widget, GdkEventButton *event,
                                                     dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  if(event->button == GDK_BUTTON_PRIMARY)
  {
    g->is_dragging = FALSE;
  }
  return TRUE;
}

static gboolean dt_iop_zonesystem_bar_scrolled(GtkWidget *widget, GdkEventScroll *event, dt_iop_module_t *self)
{
  dt_iop_zonesystem_params_t *p = self->params;
  int cs = CLAMP(p->size, 4, MAX_ZONE_SYSTEM_SIZE);

  if(dt_gui_ignore_scroll(event)) return FALSE;

  int delta_y;
  if(dt_gui_get_scroll_unit_delta(event, &delta_y))
  {
    p->size = CLAMP(p->size - delta_y, 4, MAX_ZONE_SYSTEM_SIZE);
    p->zone[cs] = -1;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
    gtk_widget_queue_draw(widget);
  }

  return TRUE;
}

static gboolean dt_iop_zonesystem_bar_leave_notify(GtkWidget *widget, GdkEventCrossing *event,
                                                   dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  g->hilite_zone = FALSE;
  gtk_widget_queue_draw(g->preview);
  return TRUE;
}

static gboolean dt_iop_zonesystem_bar_motion_notify(GtkWidget *widget, GdkEventMotion *event,
                                                    dt_iop_module_t *self)
{
  dt_iop_zonesystem_params_t *p = self->params;
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  const int inset = DT_ZONESYSTEM_INSET;
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width - 2 * inset, height = allocation.height - 2 * inset;

  /* calculate zonemap */
  float zonemap[MAX_ZONE_SYSTEM_SIZE] = { -1 };
  _iop_zonesystem_calculate_zonemap(p, zonemap,  NULL, 0);

  /* record mouse position within control */
  g->mouse_x = CLAMP(event->x - inset, 0, width);
  g->mouse_y = CLAMP(height - 1 - event->y + inset, 0, height);

  if(g->is_dragging)
  {
    if((g->mouse_x / width) > zonemap[g->current_zone - 1]
       && (g->mouse_x / width) < zonemap[g->current_zone + 1])
    {
      p->zone[g->current_zone] = (g->mouse_x / width);
      dt_dev_add_history_item(darktable.develop, self, TRUE);
    }
  }
  else
  {
    /* decide which zone the mouse is over */
    if(g->mouse_y >= height * (1.0 - DT_ZONESYSTEM_REFERENCE_SPLIT))
    {
      g->zone_under_mouse = (g->mouse_x / width) / (1.0 / (p->size - 1));
      g->mouse_over_output_zones = TRUE;
    }
    else
    {
      float xpos = g->mouse_x / width;
      for(int z = 0; z < p->size - 1; z++)
      {
        if(xpos >= zonemap[z] && xpos < zonemap[z + 1])
        {
          g->zone_under_mouse = z;
          break;
        }
      }
      g->mouse_over_output_zones = FALSE;
    }
    g->hilite_zone = (g->mouse_y < height) ? TRUE : FALSE;
  }

  gtk_widget_queue_draw(self->widget);
  gtk_widget_queue_draw(g->preview);
  return TRUE;
}


static gboolean dt_iop_zonesystem_preview_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  const int inset = DT_PIXEL_APPLY_DPI(2);
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width, height = allocation.height;

  dt_iop_zonesystem_gui_data_t *g = self->gui_data;
  dt_iop_zonesystem_params_t *p = self->params;

  cairo_surface_t *cst = dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);

  /* clear background */
  GtkStyleContext *context = gtk_widget_get_style_context(self->expander);
  gtk_render_background(context, cr, 0, 0, allocation.width, allocation.height);

  width -= 2 * inset;
  height -= 2 * inset;
  cairo_translate(cr, inset, inset);

  dt_iop_gui_enter_critical_section(self);
  if(g->in_preview_buffer && g->out_preview_buffer && self->enabled)
  {
    /* calculate the zonemap */
    float zonemap[MAX_ZONE_SYSTEM_SIZE] = { -1 };
    _iop_zonesystem_calculate_zonemap(p, zonemap, NULL, 0);

    /* let's generate a pixbuf from pixel zone buffer */
    guchar *image = g_malloc_n((size_t)4 * g->preview_width * g->preview_height, sizeof(guchar));
    guchar *buffer = g->mouse_over_output_zones ? g->out_preview_buffer : g->in_preview_buffer;
    for(int k = 0; k < g->preview_width * g->preview_height; k++)
    {
      int zone = 255 * CLIP(((1.0 / (p->size - 1)) * buffer[k]));
      image[4 * k + 2] = (g->hilite_zone && buffer[k] == g->zone_under_mouse) ? 255 : zone;
      image[4 * k + 1] = (g->hilite_zone && buffer[k] == g->zone_under_mouse) ? 255 : zone;
      image[4 * k + 0] = (g->hilite_zone && buffer[k] == g->zone_under_mouse) ? 0 : zone;
    }
    dt_iop_gui_leave_critical_section(self);

    const int wd = g->preview_width, ht = g->preview_height;
    const float scale = fminf(width / (float)wd, height / (float)ht);
    const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, wd);
    cairo_surface_t *surface = cairo_image_surface_create_for_data(image, CAIRO_FORMAT_RGB24, wd, ht, stride);
    cairo_translate(cr, width / 2.0, height / 2.0f);
    cairo_scale(cr, scale, scale);
    cairo_translate(cr, -.5f * wd, -.5f * ht);

    cairo_rectangle(cr, DT_PIXEL_APPLY_DPI(1), DT_PIXEL_APPLY_DPI(1), wd - DT_PIXEL_APPLY_DPI(2),
                    ht - DT_PIXEL_APPLY_DPI(2));
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_GOOD);
    cairo_fill_preserve(cr);
    cairo_surface_destroy(surface);

    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
    cairo_set_source_rgb(cr, .1, .1, .1);
    cairo_stroke(cr);

    g_free(image);
  }
  else
  {
    dt_iop_gui_leave_critical_section(self);
    // draw a big, subdued dt logo
    if(g->image)
    {
      GdkRGBA *color;
      gtk_style_context_get(context, gtk_widget_get_state_flags(self->expander), "background-color", &color,
                            NULL);

      cairo_set_source_surface(cr, g->image, (width - g->image_width) * 0.5, (height - g->image_height) * 0.5);
      cairo_rectangle(cr, 0, 0, width, height);
      cairo_set_operator(cr, CAIRO_OPERATOR_HSL_LUMINOSITY);
      cairo_fill_preserve(cr);
      cairo_set_operator(cr, CAIRO_OPERATOR_DARKEN);
      cairo_set_source_rgb(cr, color->red + 0.02, color->green + 0.02, color->blue + 0.02);
      cairo_fill_preserve(cr);
      cairo_set_operator(cr, CAIRO_OPERATOR_LIGHTEN);
      cairo_set_source_rgb(cr, color->red - 0.02, color->green - 0.02, color->blue - 0.02);
      cairo_fill(cr);

      gdk_rgba_free(color);
    }
  }

  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);

  return TRUE;
}

void _iop_zonesystem_redraw_preview_callback(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_zonesystem_gui_data_t *g = self->gui_data;

  dt_control_queue_redraw_widget(g->preview);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

