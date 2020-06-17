import PredefinedFolder from './model';
import { PREDEFINED_FOLDERS } from '../../utils/constant';
import { successMessage } from 'iyasunday';

export async function seed() {
  try {
    await PredefinedFolder.bulkCreate(
      PREDEFINED_FOLDERS.map(folderName => ({ name: folderName }))
    );
    return successMessage('System folders created');
  } catch (err) {
    throw err;
  }
}

export async function list() {
  try {
    const predefinedFolders = await PredefinedFolder.findAll();

    return {
      success: true,
      data: predefinedFolders,
    };
  } catch (err) {
    throw err;
  }
}
