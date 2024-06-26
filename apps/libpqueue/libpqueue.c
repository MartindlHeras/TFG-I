/*
 * Copyright (c) 2014, Volkan Yazıcı <volkan.yazici@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/**
 * @file  pqueue.h
 * @brief Priority Queue function declarations
 *
 * @{
 */


#ifndef PQUEUE_H
#define PQUEUE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

/** priority data type */
typedef unsigned long long pqueue_pri_t;

/** callback functions to get/set/compare the priority of an element */
typedef pqueue_pri_t (*pqueue_get_pri_f)(void *a);
typedef void (*pqueue_set_pri_f)(void *a, pqueue_pri_t pri);
typedef int (*pqueue_cmp_pri_f)(pqueue_pri_t next, pqueue_pri_t curr);


/** callback functions to get/set the position of an element */
typedef size_t (*pqueue_get_pos_f)(void *a);
typedef void (*pqueue_set_pos_f)(void *a, size_t pos);


/** debug callback function to print a entry */
typedef void (*pqueue_print_entry_f)(FILE *out, void *a);


/** the priority queue handle */
typedef struct pqueue_t
{
    size_t size;                /**< number of elements in this queue */
    size_t avail;               /**< slots available in this queue */
    size_t step;                /**< growth stepping setting */
    pqueue_cmp_pri_f cmppri;    /**< callback to compare nodes */
    pqueue_get_pri_f getpri;    /**< callback to get priority of a node */
    pqueue_set_pri_f setpri;    /**< callback to set priority of a node */
    pqueue_get_pos_f getpos;    /**< callback to get position of a node */
    pqueue_set_pos_f setpos;    /**< callback to set position of a node */
    void **d;                   /**< The actualy queue in binary heap form */
} pqueue_t;


/**
 * initialize the queue
 *
 * @param n the initial estimate of the number of queue items for which memory
 *     should be preallocated
 * @param cmppri The callback function to run to compare two elements
 *     This callback should return 0 for 'lower' and non-zero
 *     for 'higher', or vice versa if reverse priority is desired
 * @param setpri the callback function to run to assign a score to an element
 * @param getpri the callback function to run to set a score to an element
 * @param getpos the callback function to get the current element's position
 * @param setpos the callback function to set the current element's position
 *
 * @return the handle or NULL for insufficent memory
 */
pqueue_t *
pqueue_init(size_t n,
            pqueue_cmp_pri_f cmppri,
            pqueue_get_pri_f getpri,
            pqueue_set_pri_f setpri,
            pqueue_get_pos_f getpos,
            pqueue_set_pos_f setpos);


/**
 * free all memory used by the queue
 * @param q the queue
 */
void pqueue_free(pqueue_t *q);


/**
 * return the size of the queue.
 * @param q the queue
 */
size_t pqueue_size(pqueue_t *q);


/**
 * insert an item into the queue.
 * @param q the queue
 * @param d the item
 * @return 0 on success
 */
int pqueue_insert(pqueue_t *q, void *d);


/**
 * move an existing entry to a different priority
 * @param q the queue
 * @param new_pri the new priority
 * @param d the entry
 */
void
pqueue_change_priority(pqueue_t *q,
                       pqueue_pri_t new_pri,
                       void *d);


/**
 * pop the highest-ranking item from the queue.
 * @param q the queue
 * @return NULL on error, otherwise the entry
 */
void *pqueue_pop(pqueue_t *q);


/**
 * remove an item from the queue.
 * @param q the queue
 * @param d the entry
 * @return 0 on success
 */
int pqueue_remove(pqueue_t *q, void *d);


/**
 * access highest-ranking item without removing it.
 * @param q the queue
 * @return NULL on error, otherwise the entry
 */
void *pqueue_peek(pqueue_t *q);


/**
 * print the queue
 * @internal
 * DEBUG function only
 * @param q the queue
 * @param out the output handle
 * @param the callback function to print the entry
 */
void
pqueue_print(pqueue_t *q, 
             FILE *out,
             pqueue_print_entry_f print);


/**
 * dump the queue and it's internal structure
 * @internal
 * debug function only
 * @param q the queue
 * @param out the output handle
 * @param the callback function to print the entry
 */
void
pqueue_dump(pqueue_t *q, 
             FILE *out,
             pqueue_print_entry_f print);


/**
 * checks that the pq is in the right order, etc
 * @internal
 * debug function only
 * @param q the queue
 */
int pqueue_is_valid(pqueue_t *q);


#endif /* PQUEUE_H */
/** @} */

/*
 * Copyright (c) 2014, Volkan Yazıcı <volkan.yazici@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#define left(i)   ((i) << 1)
#define right(i)  (((i) << 1) + 1)
#define parent(i) ((i) >> 1)


pqueue_t *
pqueue_init(size_t n,
            pqueue_cmp_pri_f cmppri,
            pqueue_get_pri_f getpri,
            pqueue_set_pri_f setpri,
            pqueue_get_pos_f getpos,
            pqueue_set_pos_f setpos)
{
    pqueue_t *q;

    if (!(q = malloc(sizeof(pqueue_t))))
        return NULL;

    /* Need to allocate n+1 elements since element 0 isn't used. */
    if (!(q->d = malloc((n + 1) * sizeof(void *)))) {
        free(q);
        return NULL;
    }

    q->size = 1;
    q->avail = q->step = (n+1);  /* see comment above about n+1 */
    q->cmppri = cmppri;
    q->setpri = setpri;
    q->getpri = getpri;
    q->getpos = getpos;
    q->setpos = setpos;

    return q;
}


void
pqueue_free(pqueue_t *q)
{
    free(q->d);
    free(q);
}


size_t
pqueue_size(pqueue_t *q)
{
    /* queue element 0 exists but doesn't count since it isn't used. */
    return (q->size - 1);
}


static void
bubble_up(pqueue_t *q, size_t i)
{
    size_t parent_node;
    void *moving_node = q->d[i];
    pqueue_pri_t moving_pri = q->getpri(moving_node);

    for (parent_node = parent(i);
         ((i > 1) && q->cmppri(q->getpri(q->d[parent_node]), moving_pri));
         i = parent_node, parent_node = parent(i))
    {
        q->d[i] = q->d[parent_node];
        q->setpos(q->d[i], i);
    }

    q->d[i] = moving_node;
    q->setpos(moving_node, i);
}


static size_t
maxchild(pqueue_t *q, size_t i)
{
    size_t child_node = left(i);

    if (child_node >= q->size)
        return 0;

    if ((child_node+1) < q->size &&
        q->cmppri(q->getpri(q->d[child_node]), q->getpri(q->d[child_node+1])))
        child_node++; /* use right child instead of left */

    return child_node;
}


static void
percolate_down(pqueue_t *q, size_t i)
{
    size_t child_node;
    void *moving_node = q->d[i];
    pqueue_pri_t moving_pri = q->getpri(moving_node);

    while ((child_node = maxchild(q, i)) &&
           q->cmppri(moving_pri, q->getpri(q->d[child_node])))
    {
        q->d[i] = q->d[child_node];
        q->setpos(q->d[i], i);
        i = child_node;
    }

    q->d[i] = moving_node;
    q->setpos(moving_node, i);
}


int
pqueue_insert(pqueue_t *q, void *d)
{
    void *tmp;
    size_t i;
    size_t newsize;

    if (!q) return 1;

    /* allocate more memory if necessary */
    if (q->size >= q->avail) {
        newsize = q->size + q->step;
        if (!(tmp = realloc(q->d, sizeof(void *) * newsize)))
            return 1;
        q->d = tmp;
        q->avail = newsize;
    }

    /* insert item */
    i = q->size++;
    q->d[i] = d;
    bubble_up(q, i);

    return 0;
}


void
pqueue_change_priority(pqueue_t *q,
                       pqueue_pri_t new_pri,
                       void *d)
{
    size_t posn;
    pqueue_pri_t old_pri = q->getpri(d);

    q->setpri(d, new_pri);
    posn = q->getpos(d);
    if (q->cmppri(old_pri, new_pri))
        bubble_up(q, posn);
    else
        percolate_down(q, posn);
}


int
pqueue_remove(pqueue_t *q, void *d)
{
    size_t posn = q->getpos(d);
    q->d[posn] = q->d[--q->size];
    if (q->cmppri(q->getpri(d), q->getpri(q->d[posn])))
        bubble_up(q, posn);
    else
        percolate_down(q, posn);

    return 0;
}


void *
pqueue_pop(pqueue_t *q)
{
    void *head;

    if (!q || q->size == 1)
        return NULL;

    head = q->d[1];
    q->d[1] = q->d[--q->size];
    percolate_down(q, 1);

    return head;
}


void *
pqueue_peek(pqueue_t *q)
{
    void *d;
    if (!q || q->size == 1)
        return NULL;
    d = q->d[1];
    return d;
}


void
pqueue_dump(pqueue_t *q,
            FILE *out,
            pqueue_print_entry_f print)
{
    int i;

    fprintf(stdout,"posn\tleft\tright\tparent\tmaxchild\t...\n");
    for (i = 1; i < q->size ;i++) {
        fprintf(stdout,
                "%d\t%d\t%d\t%d\t%ul\t",
                i,
                left(i), right(i), parent(i),
                (unsigned int)maxchild(q, i));
        print(out, q->d[i]);
    }
}


// static void
// set_pos(void *d, size_t val)
// {
//     /* do nothing */
// }


// static void
// set_pri(void *d, pqueue_pri_t pri)
// {
//     /* do nothing */
// }


static int
subtree_is_valid(pqueue_t *q, int pos)
{
    if (left(pos) < q->size) {
        /* has a left child */
        if (q->cmppri(q->getpri(q->d[pos]), q->getpri(q->d[left(pos)])))
            return 0;
        if (!subtree_is_valid(q, left(pos)))
            return 0;
    }
    if (right(pos) < q->size) {
        /* has a right child */
        if (q->cmppri(q->getpri(q->d[pos]), q->getpri(q->d[right(pos)])))
            return 0;
        if (!subtree_is_valid(q, right(pos)))
            return 0;
    }
    return 1;
}


int
pqueue_is_valid(pqueue_t *q)
{
    return subtree_is_valid(q, 1);
}

/*
 * Copyright (c) 2014, Volkan Yazıcı <volkan.yazici@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/**
 * A simple usage of libpqueue. (Priorities are represented as "double" values.)
 */


#include <stdio.h>
#include <stdlib.h>


typedef struct node_t
{
	pqueue_pri_t pri;
	int    val;
	size_t pos;
} node_t;


static int
cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
	return (next < curr);
}


static pqueue_pri_t
get_pri(void *a)
{
	return ((node_t *) a)->pri;
}


static void
set_pri(void *a, pqueue_pri_t pri)
{
	((node_t *) a)->pri = pri;
}


static size_t
get_pos(void *a)
{
	return ((node_t *) a)->pos;
}


static void
set_pos(void *a, size_t pos)
{
	((node_t *) a)->pos = pos;
}

void
pqueue_print(pqueue_t *q,
             FILE *out,
             pqueue_print_entry_f print)
{
    pqueue_t *dup;
	void *e;

    dup = pqueue_init(q->size,
                      q->cmppri, q->getpri, set_pri,
                      q->getpos, set_pos);
    dup->size = q->size;
    dup->avail = q->avail;
    dup->step = q->step;

    memcpy(dup->d, q->d, (q->size * sizeof(void *)));

    while ((e = pqueue_pop(dup)))
		print(out, e);

    pqueue_free(dup);
}

int
main(int argc, char const *argv[])
{
	pqueue_t *pq;
	node_t   *ns;
	node_t   *n;

    if (argc % 2 != 1)
    {
        printf("Error with parameters\n");
        return -1;
    }

    ns = malloc(argc * sizeof(node_t));
	pq = pqueue_init(argc, cmp_pri, get_pri, set_pri, get_pos, set_pos);
	if (!(ns && pq)) return 1;
    int j=0;
    for (int i = 1; argv[i]; i+=2)
    {
        ns[j].pri=atoi(argv[i]);
        ns[j].val=atoi(argv[i+1]);
        pqueue_insert(pq, &ns[j]);
        j++;
    }

	// ns[0].pri = 5; ns[0].val = -5; pqueue_insert(pq, &ns[0]);
	// ns[1].pri = 4; ns[1].val = -4; pqueue_insert(pq, &ns[1]);
	// ns[2].pri = 2; ns[2].val = -2; pqueue_insert(pq, &ns[2]);
	// ns[3].pri = 6; ns[3].val = -6; pqueue_insert(pq, &ns[3]);
	// ns[4].pri = 1; ns[4].val = -1; pqueue_insert(pq, &ns[4]);

	n = pqueue_peek(pq);
	printf("peek: %lld [%d]\n", n->pri, n->val);

	pqueue_change_priority(pq, 8, &ns[4]);
	pqueue_change_priority(pq, 7, &ns[2]);

   	while ((n = pqueue_pop(pq)))
		printf("pop: %lld [%d]\n", n->pri, n->val);

	pqueue_free(pq);
	free(ns);

	return 0;
}

/*
 * $ cc -Wall -g pqueue.c sample.c -o sample
 * $ ./sample
 * peek: 6 [-6]
 * pop: 8 [-1]
 * pop: 7 [-2]
 * pop: 6 [-6]
 * pop: 5 [-5]
 * pop: 4 [-4]
 */
