'use strict';
module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('files_mains', {
      id: {
        type: Sequelize.UUID,
        primaryKey: true,
        defaultValue: Sequelize.UUIDV1,
      },
      folderId: {
        type: Sequelize.UUID,
        onDelete: 'cascade',
        onUpdate: 'cascade',

        references: {
            model: 'folders_mains',
            key: 'id',
            deferrable: Sequelize.Deferrable.INITIALLY_DEFERRED,
            onDelete: 'cascade',
            onUpdate: 'cascade',
        },
      },
      name: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false,
      },
      type: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false
      },
      location: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('files_mains');
  }
};