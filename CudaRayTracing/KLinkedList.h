#pragma once

#ifndef K_LINKED_LIST_H
#define K_LINKED_LIST_H

#define CUDA_FUNC __host__ __device__

template <typename T>
struct KNode {
	CUDA_FUNC KNode() {}
	CUDA_FUNC ~KNode() {}

	T data;
	KNode* next;
	typedef T value_type;
};

template <typename T2>
class KLinkedList {
public:
	// Constructors
	CUDA_FUNC KLinkedList();
	// Destructors
	CUDA_FUNC ~KLinkedList();

private:
	// Members
	KNode<T2>* head_ptr;
	int count;

public:
	CUDA_FUNC void InsertNode(const KNode<T2>* const node);
	CUDA_FUNC void DeleteNode(const T2& data);
	CUDA_FUNC int GetCount() const;
};

template <typename TNode>
class LinkedListIterator
{
	friend class KLinkedList<typename TNode::value_type>;
	TNode* p;
public:
	LinkedListIterator(TNode* p) : p(p) {}
	LinkedListIterator(const LinkedListIterator& other) : p(other.p) {}
	LinkedListIterator& operator=(LinkedListIterator other) { std::swap(p, other.p); return *this; }
	void operator++() { p = p->next; }
	void operator++(int) { p = p->next; }
	bool operator==(const LinkedListIterator& other) { return p == other.p; }
	bool operator!=(const LinkedListIterator& other) { return p != other.p; }
	const int& operator*() const { return p->data; }
	LinkedListIterator<TNode> operator+(int i)
	{
		LinkedListIterator<TNode> iter = *this;
		while (i-- > 0 && iter.p)
		{
			++iter;
		}
		return iter;
	}
};

#endif