/* Copyright 2014 Jeff Taylor
 *
 * Red-Black Tree, implemented as a list.  New entries are appended to the end
 * of the list.  The tree structure is maintained by indices instead of
 * pointers, as this will later be implemented in OpenCL, where pointers,
 * dynamic memory, and recursion are all unavailable.
 *
 * New allocations come at the end of the list.  We don't support deletion, so
 * the tree can only grow.
 *
 * We handle the lack of recursion by using an explicit stack to trace our
 * progress through the tree, which we then 'unwind' to walk back up the tree,
 * fixing violations in O(logN) time.
 *
 * Heavily inspired by Julienne Walker's Red-Black Tree tutorial:
 * http://eternallyconfuzzled.com/tuts/datastructures/jsw_tut_rbtree.aspx
 */

/* TODO(jeff): kMaxSize should *not* be hardcoded here!
 * TODO(jeff): kMaxSteps should *not* be the buffer size.  If it were possible
 * for the particle to touch a new box with each step, it would be, but this is
 * both extremely pessimistic and geometrically impossible.  Therefore, figure
 * out a more sensible limit!
 */
#define kMaxSize  2044
#define kMaxDepth 12   /* 2*log2(kMaxSize) */

#define BLACK 0
#define RED   1

#define LEAF -1

typedef int data_t;

struct rbtree_node {
  int data;
  short child[2];  /* I like Julienne Walker's binary tree idiom */
} __attribute__((aligned(8)));

struct rbtree {
  struct rbtree_node nodes[kMaxSize]; /* 2041 * 8 bytes */
  short num_entries; /* 2 bytes */
  short root; /* 2 bytes */
  short node_stack[kMaxDepth]; /* 24 bytes */
  short dir_stack[kMaxDepth]; /* 24 bytes */
  char pad[4]; /* We're 12 bytes short of a nice round 16384B */
} __attribute__((packed));  /* shouldn't affect the structure packing. */

int eq(data_t a, data_t b)
{
  return a == b;
}

int cmp(data_t a, data_t b)
{
  return a < b;
}

int is_red(global struct rbtree *tree, int node)
{
  if ((LEAF != node) && (RED == ((1<<31) & tree->nodes[node].data)))
    return 1;
  else
    return 0;
}

void rbtree_colour_red(global struct rbtree *tree, int node)
{
  tree->nodes[node].data |= (1<<31);
}

void rbtree_colour_black(global struct rbtree *tree, int node)
{
  tree->nodes[node].data &= ~(1<<31);
}

data_t rbtree_data(global struct rbtree *tree, int node)
{
  return tree->nodes[node].data & ~(1<<31);
}

void rbtree_init(global struct rbtree *tree)
{
  tree->num_entries = 0;
  tree->root = LEAF;
}

int rbtree_mknode(global struct rbtree *tree, data_t data)
{
  int new_node;

  // assert(tree->num_entries < kMaxSize);

  new_node = tree->num_entries++;
  tree->nodes[new_node].data = data;
  rbtree_colour_red(tree, new_node);
  tree->nodes[new_node].child[0] = LEAF;
  tree->nodes[new_node].child[1] = LEAF;

  return new_node;
}

/* Rotate an rbtree, returning the new root.
 *
 *    R <-root in    N <-new root
 *   / \            / \
 *  *   N      =>  R   *
 *     / \        / \
 *    *   *      *   *
 */
int rbtree_rotate_single(global struct rbtree *tree, int root, int dir)
{
  int saved = tree->nodes[root].child[!dir];

  tree->nodes[root].child[!dir] = tree->nodes[saved].child[dir];
  tree->nodes[saved].child[dir] = root;

  rbtree_colour_red(tree, root);
  rbtree_colour_black(tree, saved);

  return saved;
}

/* Rotate an rbtree twice, returning the new root.
 *
 *    R <-root in   R              B <-new root
 *  /   \          / \           /   \
 * *     A        *   B         R     A
 *      / \    =>    / \   =>  / \   / \
 *     B   *        *   A     *   * *   *
 *    / \              / \
 *   *   *            *   *
 */
int rbtree_rotate_double(global struct rbtree *tree, int root, int dir)
{
  tree->nodes[root].child[!dir]
    = rbtree_rotate_single(tree, tree->nodes[root].child[!dir], !dir);
  return rbtree_rotate_single(tree, root, dir);
}

/* Returns: 1 if there may be another violation, 0 if there are no more
 * violations possible.
 */
int rbtree_fix(global struct rbtree *tree, int stack_pos)
{
  /* At this point:
    *   node_stack[stack_pos-1] = parent
    *   node_stack[stack_pos-2] = grandparent
    *
    * A red violation exists if both me and parent are RED.  This cannot happen
    * if parent is the root (root is black), therefore, the last possible
    * violation is when the grandparent is root.
    */

  /* Check violations here */
  int me = tree->node_stack[stack_pos];

  int parent = tree->node_stack[stack_pos-1];
  int p_dir = tree->dir_stack[stack_pos-1];

  int grandparent = tree->node_stack[stack_pos-2];
  int gp_dir = tree->dir_stack[stack_pos-2];

  int uncle = tree->nodes[grandparent].child[!gp_dir];

  int ggp, ggp_dir;

  if (is_red(tree, me) && is_red(tree, parent))
  {
    /* Gasp!  A red violation! */
    if (is_red(tree, uncle))
    {
      /* Uncle is red.  Can solve with recolouring */
      rbtree_colour_red(tree, grandparent);
      rbtree_colour_black(tree, parent);
      rbtree_colour_black(tree, uncle);

      /* Make sure the root is always black */
      rbtree_colour_black(tree, tree->root);
    } else {
      /* Uncle is black.  Solve with rotations.
        * ie:
        *       GP,B          P,B
        *       / \          /  \
        *     P,R  U,B  => M,R  GP,R
        *    /                    \
        *   M,R                    U,B
        */

      if (grandparent == tree->root) {
        if (p_dir == gp_dir)
          tree->root = rbtree_rotate_single(tree, tree->root, !gp_dir);
        else
          tree->root = rbtree_rotate_double(tree, tree->root, !gp_dir);
      } else {
        ggp = tree->node_stack[stack_pos-3];
        ggp_dir = tree->dir_stack[stack_pos-3];

        if (p_dir == gp_dir) {
          tree->nodes[ggp].child[ggp_dir]
            = rbtree_rotate_single(tree, grandparent, !gp_dir);
        } else {
          tree->nodes[ggp].child[ggp_dir]
            = rbtree_rotate_double(tree, grandparent, !gp_dir);
        }
      }

      /* Now that we've done our rotation, the stack is going to be somewhat
       * garbled.  But since the new GP (now parent) is black, there's no
       * possibility of creating a new red violation.  So we can simply break
       * out of the loop.
       */
      return 0;  /* break */
    }
  }

  return 1;  /* continue */
}

/* Insert an element into the tree */
void rbtree_insert(global struct rbtree *tree, data_t data)
{
  int cur_node;
  int dir;  /* 0 = left, 1 = right */
  int stack_pos;

  /* Check if this is the first node added. */
  if (LEAF == tree->root) {
    tree->root = rbtree_mknode(tree, data);
    rbtree_colour_black(tree, tree->root);
    return;
  }

  /* Traverse the list to figure out where we belong */
  cur_node = tree->root;
  for (stack_pos = 0; stack_pos < kMaxDepth; ++stack_pos)
  {
    if (eq(rbtree_data(tree, cur_node), data))
      return;  /* duplicates ignored */

    dir = cmp(rbtree_data(tree, cur_node), data);

    /* Put the node and direction onto the stack */
    tree->node_stack[stack_pos] = cur_node;
    tree->dir_stack[stack_pos] = dir;

    if (LEAF == tree->nodes[cur_node].child[dir]) {
      /* No node.  Insert here. */
      tree->nodes[cur_node].child[dir] = rbtree_mknode(tree, data);
      break;
    }

    /* Move onto the next node. */
    cur_node = tree->nodes[cur_node].child[dir];
  }

  /* Put the newly added node on the stack */
  stack_pos++;
  tree->node_stack[stack_pos] = cur_node;

  /* Walk back up the stack, fixing violations as we go */
  for(; stack_pos >= 2; --stack_pos) {
    if (!rbtree_fix(tree, stack_pos))
      break;  /* No more violations possible */
  }
}

