'use strict';
import Folder from './model';
import { NotFoundError, successMessage, slugify, paginate } from 'iyasunday';

export async function create(user, body) {
  try {
    body.userId = user.id;
    body.slug = slugify(body.name);
    const folder = await Folder.create(body);
    return {
      success: true,
      data: folder,
    };
  } catch (err) {
    throw err;
  }
}

export async function update(user, { id }, body) {
  try {
    let folder = await Folder.findByPk(id);
    if (!folder) throw new NotFoundError('Folder not found');
    folder = await folder.update(body);
    return {
      success: true,
      data: folder,
    };
  } catch (err) {
    throw err;
  }
}

export async function list({ id: userId }, { predefinedFolderId, pageId, limit }) {
  try {
    const where = { userId, predefinedFolderId };
    const totalCount = await Folder.count({ where });
    const { offset, pageCount } = paginate(totalCount, pageId);
    const folders = await Folder.findAll({
      where,
      offset,
      limit: parseInt(limit),
    });
    return {
      success: true,
      data: {
        folders,
        limit,
        totalCount,
        pageCount,
        pageId,
      },
    };
  } catch (err) {
    throw err;
  }
}

export async function view({ id: userId }, { id }) {
  try {
    const folder = await Folder.findOne({ where: { userId, id } });
    if (!folder) throw new NotFoundError('Folder not found');
    return {
      success: true,
      data: folder,
    };
  } catch (err) {
    throw err;
  }
}

export async function remove({ id: userId }, { id }) {
  try {
    const folder = await Folder.findOne({ where: { userId, id } });
    if (!folder) throw new NotFoundError('Folder not found');
    await folder.destroy();
    return successMessage(folder.name + ' deleted');
  } catch (err) {
    throw err;
  }
}
