'use strict';
import File from './model';
import {
  NotFoundError,
  successMessage,
  slugify,
  paginate,
  deleteFile,
  fileExists,
  ExistsError
} from 'iyasunday';

export async function create({folderId},body,uploadedFile) {
  try {
    body.slug = slugify(body.name);
    body.folderId=folderId;
    body.file = uploadedFile.filename;
    let file = await File.findOne({where:{folderId, name:body.name}});
    if(file) throw new ExistsError("File name already exists");
    file = await File.create(body);
    return {
      success : true,
      data : file
    }
  } catch (err) {
    throw err;
  }
}

export async function update({id},body,uploadedFile) {
  try {
    let file = await File.findByPk(id);
    if(!file) throw new NotFoundError("File not found");
    if(await fileExists(process.env.STORAGE_PATH+'/'+file.file))
      await deleteFile(process.env.STORAGE_PATH+'/'+file.file);
    body.file = uploadedFile.filename;
    file = await file.update(body);
    return {
      success : true,
      data : file
    }
  } catch (err) {
    throw err;
  }
}



export async function list({folderId},{pageId,limit}) {
  try {
    const totalCount = await File.count({where:{folderId}});
    const {offset,pageCount} = paginate(totalCount,pageId);
    const files = await File.findAll({where:{folderId}, offset, limit:parseInt(limit)});
    return {
      success : true,
      data : {
        files,
        limit,
        totalCount,
        pageCount,
        pageId
      }
    }
  } catch (err) {
    throw err;
  }
}

export async function view({id}) {
  try {
    const file = await File.findByPk(id);
    if(!file) throw new NotFoundError("File not found");
    return {
      success: true,
      data: file,
    };
  } catch (err) {
    throw err;
  }
}

export async function remove({id}) {
  try {
    const file = await File.findByPk(id);
    if(!file) throw new NotFoundError("File not found");
    await file.destroy();
    return successMessage(file.name+' deleted')
  } catch (err) {
    throw err;
  }
}

